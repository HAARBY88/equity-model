"""
Stock Screener — universe.py + screener.py combined
Scans S&P 100 universe, pre-filters, then runs the full 5-layer
statistical model and outputs a ranked shortlist.

Run with: python screener.py
Dependencies: pip install yfinance pandas numpy requests
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── UNIVERSE DEFINITION ───────────────────────────────────────────────────────

# S&P 100 tickers (most liquid US large caps)
SP100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","TSLA","LLY",
    "JPM","UNH","XOM","V","AVGO","PG","MA","JNJ","HD","COST",
    "MRK","ABBV","CVX","NFLX","BAC","KO","PEP","TMO","WMT","ADBE",
    "CRM","CSCO","MCD","ACN","ABT","ORCL","AMD","IBM","QCOM","TXN",
    "LIN","GE","PM","DHR","NEE","INTU","NOW","CAT","SPGI","BA",
    "RTX","UNP","GS","MS","T","LOW","ISRG","ELV","MDT","AMGN",
    "BLK","DE","SYK","AXP","GILD","REGN","SCHW","PLD","C","CB",
    "ADI","MMC","VRTX","ETN","SO","DUK","MO","ZTS","BSX","BDX",
    "MDLZ","CI","CL","ITW","WM","AON","CSX","EW","APD","PGR",
    "CME","NSC","MCO","USB","EMR","FDX","ECL","TGT","SHW","ICE"
]

# Sector assignments for S&P 100
SECTOR_MAP = {
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","GOOGL":"XLC","GOOG":"XLC",
    "META":"XLC","NFLX":"XLC","ADBE":"XLK","CRM":"XLK","AMD":"XLK",
    "CSCO":"XLK","IBM":"XLK","QCOM":"XLK","TXN":"XLK","ORCL":"XLK",
    "ACN":"XLK","INTU":"XLK","NOW":"XLK","ADI":"XLK","AVGO":"XLK",
    "AMZN":"XLY","TSLA":"XLY","HD":"XLY","MCD":"XLY","LOW":"XLY",
    "TGT":"XLY","NKE":"XLY",
    "WMT":"XLP","PG":"XLP","KO":"XLP","PEP":"XLP","COST":"XLP",
    "MO":"XLP","PM":"XLP","CL":"XLP","MDLZ":"XLP",
    "JPM":"XLF","BAC":"XLF","GS":"XLF","MS":"XLF","V":"XLF",
    "MA":"XLF","AXP":"XLF","BLK":"XLF","SCHW":"XLF","C":"XLF",
    "CB":"XLF","MMC":"XLF","CME":"XLF","AON":"XLF","MCO":"XLF",
    "USB":"XLF","PGR":"XLF","ICE":"XLF","SPGI":"XLF",
    "XOM":"XLE","CVX":"XLE",
    "UNH":"XLV","JNJ":"XLV","LLY":"XLV","MRK":"XLV","ABBV":"XLV",
    "ABT":"XLV","TMO":"XLV","DHR":"XLV","MDT":"XLV","AMGN":"XLV",
    "GILD":"XLV","REGN":"XLV","ISRG":"XLV","SYK":"XLV","BDX":"XLV",
    "ELV":"XLV","CI":"XLV","BSX":"XLV","ZTS":"XLV","EW":"XLV","VRTX":"XLV",
    "CAT":"XLI","BA":"XLI","RTX":"XLI","UNP":"XLI","GE":"XLI",
    "DE":"XLI","ETN":"XLI","NSC":"XLI","CSX":"XLI","EMR":"XLI",
    "ITW":"XLI","WM":"XLI","FDX":"XLI",
    "NEE":"XLU","SO":"XLU","DUK":"XLU",
    "PLD":"XLRE","BRK-B":"XLF",
    "LIN":"XLB","APD":"XLB","ECL":"XLB","SHW":"XLB",
    "T":"XLC","VZ":"XLC",
}

WEIGHTS = {
    "price_vol":0.30,"sector_rs":0.25,"breadth":0.20,
    "volatility":0.15,"yield_curve":0.10
}

# ── SCREENING PARAMETERS ──────────────────────────────────────────────────────

# Pre-filter rules
MIN_AVG_VOLUME   = 1_000_000   # avg daily volume
MIN_PRICE        = 10.0        # minimum stock price
REQUIRE_UPTREND  = True        # price > 50MA
REQUIRE_MOMENTUM = True        # 20-day return > 0

# Statistical model thresholds
MIN_WIN_PROB     = 0.55
MIN_Z            = 0.5
MAX_Z            = 2.5
KELLY_FRAC       = 0.5
MAX_POSITION     = 0.15
MIN_EV           = 0           # minimum expected value $
P25_FLOOR        = -500        # worst acceptable P25 loss
MONTE_N          = 500         # MC runs (lower = faster for screening)
TOP_N            = 10          # return top N stocks

LOOKBACK         = 420         # calendar days of history

# ── HELPERS ───────────────────────────────────────────────────────────────────

def normalise(val, lo, hi):
    return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))

def get_col(raw, ticker, field):
    try:
        if (ticker, field) in raw.columns: return raw[(ticker, field)].dropna()
        if field in raw.columns:           return raw[field].dropna()
        return pd.Series(dtype=float)
    except: return pd.Series(dtype=float)

def compute_dms(c, v, vix_s, spy_s, sc, t10, t5):
    avg_vol   = v.rolling(20).mean().iloc[-1]
    vol_ratio = v.iloc[-1]/avg_vol if avg_vol>0 else 1.0
    pv        = normalise((c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*vol_ratio,-0.05,0.05)
    vix_score = 100-normalise(vix_s.iloc[-1],10,40)
    hi=c.rolling(2).max(); lo=c.rolling(2).min()
    atr_pct   = ((hi-lo)/c).rolling(14).mean().iloc[-1]*100
    vlt       = (vix_score+(100-normalise(atr_pct,0.5,5.0)))/2
    brd       = normalise(spy_s.iloc[-1]/spy_s.rolling(50).mean().iloc[-1],0.90,1.10)
    try:    yld=normalise(t10.iloc[-1]-t5.iloc[-1],-0.5,2.0)
    except: yld=50.0
    w=min(20,len(c)-1)
    sr=(c.iloc[-1]/c.iloc[-w])-1; er=(sc.iloc[-1]/sc.iloc[-w])-1
    rs=normalise((1+sr)/((1+er) if (1+er)!=0 else 1),0.70,1.30)
    return round(WEIGHTS["price_vol"]*pv+WEIGHTS["sector_rs"]*rs+
                 WEIGHTS["breadth"]*brd+WEIGHTS["volatility"]*vlt+
                 WEIGHTS["yield_curve"]*yld,2)

def bayesian_win_prob(c, dms_now, window=20, horizon=10):
    ret=c.pct_change().dropna()
    rm=ret.rolling(window).mean(); rs=ret.rolling(window).std()
    proxy=((rm/rs.replace(0,np.nan)).clip(-3,3)*16.67+50).dropna()
    fwd=c.pct_change(horizon).shift(-horizon)
    aligned=pd.concat([proxy,fwd],axis=1).dropna(); aligned.columns=["d","f"]
    lo=max(0,dms_now-10); hi=min(100,dms_now+10)
    b=aligned[(aligned["d"]>=lo)&(aligned["d"]<=hi)]
    if len(b)<10: return 0.50, len(b)
    return round(float((b["f"]>0).mean()),3), len(b)

def zscore_entry(c, window=20):
    ret=c.pct_change().dropna()
    mu=ret.rolling(window).mean().iloc[-1]; sd=ret.rolling(window).std().iloc[-1]
    if sd==0 or pd.isna(sd): return 0.0
    return round(float((ret.iloc[-1]-mu)/sd),2)

def kelly_size(win_prob, avg_win=0.03, avg_loss=0.015):
    if avg_loss==0: return 0.0
    b=avg_win/avg_loss; q=1-win_prob
    return round(min(max((b*win_prob-q)/b*KELLY_FRAC,0),MAX_POSITION),4)

def monte_carlo(c, kelly, capital=100000, horizon=10, n=MONTE_N):
    ret=c.pct_change().dropna().values
    entry=c.iloc[-1]; shares=(capital*kelly)/entry
    outcomes=[((entry*np.prod(1+np.random.choice(ret,size=horizon,replace=True)))-entry)*shares
              for _ in range(n)]
    oc=np.array(outcomes)
    return (round(float(np.percentile(oc,25)),0),
            round(float(np.percentile(oc,50)),0),
            round(float(np.percentile(oc,75)),0),
            round(float((oc<0).mean()),3))

def classify_regime(vix_s, spy_s):
    vix=vix_s.iloc[-1]; spy=spy_s.iloc[-1]; ma200=spy_s.rolling(200).mean().iloc[-1]
    above=spy>ma200
    if above and vix<15:  return "Bull quiet",    True
    if above and vix<25:  return "Bull volatile",  True
    if not above and vix<20: return "Bear quiet",  False
    return "Bear volatile", False

# ── STAGE 1: FETCH UNIVERSE DATA ──────────────────────────────────────────────

def fetch_universe(tickers):
    all_tix = list(set(tickers + list(set(SECTOR_MAP.values())) +
                       ["SPY","^VIX","^TNX","^FVX"]))
    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK)
    print(f"  Fetching {len(all_tix)} tickers ({start.date()} → {end.date()})...")
    raw = yf.download(all_tix, start=start, end=end,
                      auto_adjust=True, progress=False, group_by="ticker")
    return raw

# ── STAGE 2: PRE-FILTER ───────────────────────────────────────────────────────

def pre_filter(tickers, raw, vix_s, spy_s):
    """Fast rules-based filter. Returns list of tickers that pass."""
    passed = []; failed = {}

    # Regime check first — if bear market, nothing passes
    _, tradeable = classify_regime(vix_s, spy_s)
    if not tradeable:
        print("  ⚠️  Bear market regime — pre-filter blocked all stocks.")
        return [], {}

    for ticker in tickers:
        c = get_col(raw, ticker, "Close")
        v = get_col(raw, ticker, "Volume")
        if len(c) < 55 or len(v) < 20:
            failed[ticker] = "Insufficient data"; continue

        price = c.iloc[-1]
        avg_vol = v.rolling(20).mean().iloc[-1]
        ma50    = c.rolling(50).mean().iloc[-1]
        ret_20  = (c.iloc[-1] / c.iloc[-20]) - 1

        if price < MIN_PRICE:
            failed[ticker] = f"Price ${price:.2f} < ${MIN_PRICE}"; continue
        if avg_vol < MIN_AVG_VOLUME:
            failed[ticker] = f"Volume {avg_vol/1e6:.1f}M < {MIN_AVG_VOLUME/1e6:.0f}M"; continue
        if REQUIRE_UPTREND and price < ma50:
            failed[ticker] = f"Below 50MA ({price:.0f} < {ma50:.0f})"; continue
        if REQUIRE_MOMENTUM and ret_20 <= 0:
            failed[ticker] = f"Negative 20d return ({ret_20*100:.1f}%)"; continue

        passed.append(ticker)

    return passed, failed

# ── STAGE 3: STATISTICAL MODEL ────────────────────────────────────────────────

def score_ticker(ticker, raw, vix_s, spy_s, t10, t5):
    c  = get_col(raw, ticker, "Close")
    v  = get_col(raw, ticker, "Volume")
    sc = get_col(raw, SECTOR_MAP.get(ticker,"SPY"), "Close")
    if len(c) < 60: return None

    dms      = compute_dms(c, v, vix_s, spy_s, sc, t10, t5)
    win_prob, n = bayesian_win_prob(c, dms)
    z        = zscore_entry(c)
    kelly    = kelly_size(win_prob)
    p25,p50,p75,prob_loss = monte_carlo(c, kelly) if kelly>0 else (0,0,0,1.0)

    # All 5 layers
    l1 = win_prob >= MIN_WIN_PROB
    _, l2 = classify_regime(vix_s, spy_s)
    l3 = MIN_Z <= z <= MAX_Z
    l4 = kelly > 0.01
    l5 = p25 > P25_FLOOR
    all_pass = l1 and l2 and l3 and l4 and l5

    atr = (c.rolling(2).max()-c.rolling(2).min()).rolling(14).mean().iloc[-1]

    return {
        "Ticker":    ticker,
        "Price":     round(float(c.iloc[-1]), 2),
        "DMS":       dms,
        "Win Prob":  round(win_prob, 3),
        "Z-Score":   z,
        "Kelly %":   round(kelly*100, 1),
        "P25 $":     p25,
        "P50 $":     p50,
        "P75 $":     p75,
        "P(loss)":   round(prob_loss*100, 1),
        "EV $":      p50,
        "All pass":  all_pass,
        "L1 WinP":   "✅" if l1 else "❌",
        "L3 Z":      "✅" if l3 else "❌",
        "L4 Kelly":  "✅" if l4 else "❌",
        "L5 MC":     "✅" if l5 else "❌",
        "Stop":      round(float(c.iloc[-1])-1.5*float(atr), 2),
        "Sector ETF":SECTOR_MAP.get(ticker,"—"),
        "n_samples": n,
    }

# ── MAIN SCREENER ─────────────────────────────────────────────────────────────

def run_screener(universe=None, top_n=TOP_N):
    if universe is None:
        universe = SP100

    print(f"\n{'='*65}")
    print(f"  STOCK SCREENER  |  {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Universe: {len(universe)} stocks  →  target top {top_n}")
    print(f"{'='*65}\n")

    # Fetch
    print("Stage 1 — Fetching universe data...")
    raw   = fetch_universe(universe)
    vix_s = get_col(raw,"^VIX","Close"); spy_s=get_col(raw,"SPY","Close")
    t10   = get_col(raw,"^TNX","Close"); t5   =get_col(raw,"^FVX","Close")

    regime, tradeable = classify_regime(vix_s, spy_s)
    print(f"  Regime: {regime}  |  Tradeable: {'Yes ✅' if tradeable else 'No ❌'}\n")

    if not tradeable:
        print("  Bear market regime active. No trades recommended.")
        return pd.DataFrame()

    # Pre-filter
    print("Stage 2 — Pre-filtering...")
    passed, failed = pre_filter(universe, raw, vix_s, spy_s)
    print(f"  Passed: {len(passed)} / {len(universe)} stocks")
    print(f"  Filtered out: {len(failed)} stocks\n")

    if not passed:
        print("  No stocks passed pre-filter.")
        return pd.DataFrame()

    # Statistical model
    print(f"Stage 3 — Running 5-layer statistical model on {len(passed)} stocks...")
    results = []
    for i, ticker in enumerate(passed):
        try:
            r = score_ticker(ticker, raw, vix_s, spy_s, t10, t5)
            if r:
                results.append(r)
                status = "✅" if r["All pass"] else "  "
                if i % 10 == 0 or r["All pass"]:
                    print(f"  {status} {ticker:<6} DMS:{r['DMS']:>5}  "
                          f"WinP:{r['Win Prob']*100:.1f}%  "
                          f"Z:{r['Z-Score']:>5}  EV:${r['EV $']:>6}")
        except Exception as e:
            pass

    if not results:
        print("No results generated.")
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("EV $", ascending=False)

    # ── Print leaderboard ──
    trades = df[df["All pass"]]
    others = df[~df["All pass"]].head(10)

    print(f"\n{'='*65}")
    print(f"  TOP TRADE CANDIDATES TODAY — all 5 layers pass ({len(trades)})")
    print(f"{'='*65}")
    if len(trades):
        cols = ["Ticker","Price","DMS","Win Prob","Z-Score",
                "Kelly %","P25 $","EV $","Stop"]
        print(trades[cols].to_string(index=False))
    else:
        print("  None today — no stocks pass all 5 layers.")

    print(f"\n  NEAR MISSES (high EV, 1-2 layers failing):")
    print(f"  {'Ticker':<7} {'EV':>6}  Layers failing")
    for _, r in others.iterrows():
        fails = []
        if r["L1 WinP"]=="❌": fails.append("L1 WinP")
        if r["L3 Z"]=="❌":    fails.append("L3 Z-score")
        if r["L4 Kelly"]=="❌":fails.append("L4 Kelly")
        if r["L5 MC"]=="❌":   fails.append("L5 MC")
        if fails:
            print(f"  {r['Ticker']:<7} ${r['EV $']:>5}  {', '.join(fails)}")

    print(f"\n{'='*65}\n")

    # Save
    fname = f"screener_{datetime.today().strftime('%Y%m%d')}.csv"
    df.to_csv(fname, index=False)
    print(f"  Full results saved → {fname}")

    return trades.head(top_n)


# ── CUSTOM UNIVERSE HELPERS ───────────────────────────────────────────────────

def screen_sector(sector_etf, top_n=5):
    """Screen only stocks in a specific sector."""
    sector_tix = [t for t,s in SECTOR_MAP.items() if s==sector_etf]
    print(f"\nScreening {sector_etf} sector ({len(sector_tix)} stocks)...")
    return run_screener(universe=sector_tix, top_n=top_n)

def screen_custom(tickers, top_n=5):
    """Screen a custom list."""
    return run_screener(universe=tickers, top_n=top_n)


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Full S&P 100 scan
    top_trades = run_screener()

    # Or screen a single sector:
    # top_trades = screen_sector("XLK")   # Tech only
    # top_trades = screen_sector("XLF")   # Financials only

    # Or a custom list:
    # top_trades = screen_custom(["AAPL","MSFT","NVDA","TSLA","AMD"])
