"""
Equity Quant Model — Statistical Decision Engine
5 statistical layers:
  1. Signal confidence (Bayesian win probability)
  2. Regime classification (4-state market model)
  3. Z-score entry timing
  4. Kelly criterion position sizing
  5. Monte Carlo trade simulation
Run with: python model.py
Dependencies: pip install yfinance pandas numpy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

WATCHLIST = ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","JPM","XOM","UNH","V"]

SECTOR_MAP = {
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","GOOGL":"XLC","META":"XLC",
    "AMZN":"XLY","JPM":"XLF","XOM":"XLE","UNH":"XLV","V":"XLF",
}

WEIGHTS = {
    "price_vol":0.30,"sector_rs":0.25,"breadth":0.20,
    "volatility":0.15,"yield_curve":0.10
}

# Statistical thresholds
MIN_WIN_PROB    = 0.55   # Layer 1: minimum Bayesian win probability
MIN_ZSCORE      = 0.5    # Layer 3: minimum momentum z-score
MAX_ZSCORE      = 2.5    # Layer 3: maximum (overbought) z-score
KELLY_FRACTION  = 0.5    # Use half-Kelly for safety
MAX_POSITION    = 0.15   # Cap position at 15% of portfolio
MONTE_CARLO_N   = 1000   # Simulations per trade
MIN_MC_PCTILE   = 25     # 25th percentile must be acceptable
LOOKBACK_DAYS   = 252    # 1 year for probability calibration

# ── HELPERS ───────────────────────────────────────────────────────────────────

def normalise(val, lo, hi):
    return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))

def fetch(tickers, days=LOOKBACK_DAYS + 60):
    end = datetime.today()
    start = end - timedelta(days=days)
    return yf.download(list(tickers), start=start, end=end,
                       auto_adjust=True, progress=False, group_by="ticker")

def get_col(raw, ticker, field):
    try:
        if (ticker, field) in raw.columns:
            return raw[(ticker, field)].dropna()
        if field in raw.columns:
            return raw[field].dropna()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

# ── DMS ENGINE ────────────────────────────────────────────────────────────────

def compute_dms(close_s, vol_s, vix_s, spy_s, sect_s, t10, t5):
    avg_vol   = vol_s.rolling(20).mean().iloc[-1]
    vol_ratio = vol_s.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
    pct_chg   = (close_s.iloc[-1] - close_s.iloc[-2]) / close_s.iloc[-2]
    pv        = normalise(pct_chg * vol_ratio, -0.05, 0.05)
    vix_score = 100 - normalise(vix_s.iloc[-1], 10, 40)
    hi = close_s.rolling(2).max(); lo = close_s.rolling(2).min()
    atr_pct   = ((hi-lo)/close_s).rolling(14).mean().iloc[-1]*100
    vlt       = (vix_score + (100 - normalise(atr_pct, 0.5, 5.0))) / 2
    brd       = normalise(spy_s.iloc[-1]/spy_s.rolling(50).mean().iloc[-1], 0.90, 1.10)
    try:    yld = normalise(t10.iloc[-1]-t5.iloc[-1], -0.5, 2.0)
    except: yld = 50.0
    w  = min(20, len(close_s)-1)
    sr = (close_s.iloc[-1]/close_s.iloc[-w])-1
    er = (sect_s.iloc[-1]/sect_s.iloc[-w])-1
    rs = normalise((1+sr)/((1+er) if (1+er)!=0 else 1), 0.70, 1.30)
    return round(WEIGHTS["price_vol"]*pv + WEIGHTS["sector_rs"]*rs +
                 WEIGHTS["breadth"]*brd  + WEIGHTS["volatility"]*vlt +
                 WEIGHTS["yield_curve"]*yld, 2)

# ── LAYER 1: BAYESIAN WIN PROBABILITY ─────────────────────────────────────────

def bayesian_win_prob(close_s, dms_now, window=20, horizon=10):
    """
    P(profit | DMS bucket) estimated from rolling historical data.
    For each past day, compute what DMS 'would have been' (using
    recent momentum proxy) and whether the next `horizon` days
    produced a gain. Group by DMS bucket and return win rate.
    """
    returns     = close_s.pct_change().dropna()
    roll_mean   = returns.rolling(window).mean()
    roll_std    = returns.rolling(window).std()
    # Proxy historical DMS using rolling momentum strength
    proxy_dms   = ((roll_mean / roll_std.replace(0, np.nan))
                   .clip(-3, 3) * 16.67 + 50).dropna()

    forward_ret = close_s.pct_change(horizon).shift(-horizon)
    aligned     = pd.concat([proxy_dms, forward_ret], axis=1).dropna()
    aligned.columns = ["dms_proxy","fwd_ret"]

    # DMS bucket for today (±10 band)
    lo = max(0,  dms_now - 10)
    hi = min(100, dms_now + 10)
    bucket = aligned[(aligned["dms_proxy"] >= lo) & (aligned["dms_proxy"] <= hi)]

    if len(bucket) < 10:
        return 0.50, len(bucket)  # insufficient data → neutral

    win_rate = (bucket["fwd_ret"] > 0).mean()
    return round(float(win_rate), 3), len(bucket)

# ── LAYER 2: REGIME CLASSIFICATION ────────────────────────────────────────────

def classify_regime(vix_s, spy_s):
    """
    4-state regime based on VIX level and SPY vs 200MA.
    Returns regime name, tradeable flag, and historical avg monthly return.
    """
    vix_now  = vix_s.iloc[-1]
    spy_now  = spy_s.iloc[-1]
    spy_200  = spy_s.rolling(200).mean().iloc[-1]
    above_ma = spy_now > spy_200

    if above_ma and vix_now < 15:
        return "Bull quiet",    True,  "+2.1%/mo"
    elif above_ma and vix_now < 25:
        return "Bull volatile", True,  "+0.8%/mo"
    elif not above_ma and vix_now < 20:
        return "Bear quiet",    False, "-0.3%/mo"
    else:
        return "Bear volatile", False, "-2.4%/mo"

# ── LAYER 3: Z-SCORE ENTRY TIMING ─────────────────────────────────────────────

def zscore_entry(close_s, window=20):
    """
    Z-score of today's return vs rolling distribution.
    0.5–2.5 = ideal entry zone (trending but not overbought).
    """
    returns   = close_s.pct_change().dropna()
    today_ret = returns.iloc[-1]
    mean_ret  = returns.rolling(window).mean().iloc[-1]
    std_ret   = returns.rolling(window).std().iloc[-1]
    if std_ret == 0 or pd.isna(std_ret):
        return 0.0
    z = (today_ret - mean_ret) / std_ret
    return round(float(z), 2)

# ── LAYER 4: KELLY CRITERION ──────────────────────────────────────────────────

def kelly_size(win_prob, avg_win=0.03, avg_loss=0.015):
    """
    Full Kelly: f = (b*p - q) / b
    where b = win/loss ratio, p = win prob, q = 1-p
    Returns half-Kelly, capped at MAX_POSITION.
    """
    if avg_loss == 0: return 0.0
    b = avg_win / avg_loss
    p = win_prob
    q = 1 - p
    full_kelly = (b * p - q) / b
    half_kelly = full_kelly * KELLY_FRACTION
    return round(min(max(half_kelly, 0), MAX_POSITION), 4)

# ── LAYER 5: MONTE CARLO ──────────────────────────────────────────────────────

def monte_carlo(close_s, position_pct, capital=100000, horizon=10, n=MONTE_CARLO_N):
    """
    Simulate N trade outcomes using bootstrapped daily returns.
    Returns: p25 outcome $, p50 outcome $, p75 outcome $, P(loss).
    """
    returns    = close_s.pct_change().dropna().values
    entry_px   = close_s.iloc[-1]
    pos_value  = capital * position_pct
    shares     = pos_value / entry_px

    outcomes = []
    for _ in range(n):
        sampled_rets = np.random.choice(returns, size=horizon, replace=True)
        final_px     = entry_px * np.prod(1 + sampled_rets)
        pnl          = (final_px - entry_px) * shares
        outcomes.append(pnl)

    outcomes  = np.array(outcomes)
    p25       = round(float(np.percentile(outcomes, 25)), 0)
    p50       = round(float(np.percentile(outcomes, 50)), 0)
    p75       = round(float(np.percentile(outcomes, 75)), 0)
    prob_loss = round(float((outcomes < 0).mean()), 3)

    return p25, p50, p75, prob_loss

# ── DECISION ENGINE ───────────────────────────────────────────────────────────

def evaluate_trade(ticker, close_s, vol_s, vix_s, spy_s, sect_s, t10, t5):
    """
    Run all 5 layers. Return full statistical breakdown and final decision.
    """
    dms = compute_dms(close_s, vol_s, vix_s, spy_s, sect_s, t10, t5)

    # Layer 1
    win_prob, sample_n = bayesian_win_prob(close_s, dms)
    l1_pass = win_prob >= MIN_WIN_PROB

    # Layer 2
    regime, regime_tradeable, regime_ret = classify_regime(vix_s, spy_s)
    l2_pass = regime_tradeable

    # Layer 3
    z = zscore_entry(close_s)
    l3_pass = MIN_ZSCORE <= z <= MAX_ZSCORE

    # Layer 4
    kelly = kelly_size(win_prob)
    l4_pass = kelly > 0.01  # meaningful size

    # Layer 5
    p25, p50, p75, prob_loss = monte_carlo(close_s, kelly) if kelly > 0 else (0,0,0,1.0)
    l5_pass = p25 > -500  # 25th pct loss within tolerance

    all_pass = l1_pass and l2_pass and l3_pass and l4_pass and l5_pass
    ev       = round(p50, 0)

    return {
        "Ticker":      ticker,
        "Price":       round(close_s.iloc[-1], 2),
        "DMS":         dms,
        "Win Prob":    f"{win_prob*100:.1f}%",
        "L1 ✓":        "✅" if l1_pass else "❌",
        "Regime":      regime,
        "L2 ✓":        "✅" if l2_pass else "❌",
        "Z-Score":     z,
        "L3 ✓":        "✅" if l3_pass else "❌",
        "Kelly %":     f"{kelly*100:.1f}%",
        "L4 ✓":        "✅" if l4_pass else "❌",
        "P25 $":       p25,
        "P50 $":       p50,
        "P75 $":       p75,
        "P(loss)":     f"{prob_loss*100:.1f}%",
        "L5 ✓":        "✅" if l5_pass else "❌",
        "TRADE":       "✅ TRADE" if all_pass else "❌ SKIP",
        "EV $":        ev,
        "Sample N":    sample_n,
        "Regime Ret":  regime_ret,
    }

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_model():
    print(f"\n{'='*70}")
    print(f"  STATISTICAL EQUITY MODEL  |  {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    all_tix = list(set(WATCHLIST + list(SECTOR_MAP.values()) +
                       ["SPY","^VIX","^TNX","^FVX"]))
    print("Fetching data...")
    raw = fetch(all_tix)

    vix_s = get_col(raw,"^VIX","Close"); spy_s=get_col(raw,"SPY","Close")
    t10   = get_col(raw,"^TNX","Close"); t5   =get_col(raw,"^FVX","Close")

    regime, tradeable, ret = classify_regime(vix_s, spy_s)
    print(f"Regime: {regime} ({ret})  |  Tradeable: {'Yes' if tradeable else 'No'}\n")

    results = []
    for ticker in WATCHLIST:
        close_s = get_col(raw,ticker,"Close")
        vol_s   = get_col(raw,ticker,"Volume")
        sect_s  = get_col(raw,SECTOR_MAP.get(ticker,"SPY"),"Close")
        if len(close_s) < 60: continue
        try:
            r = evaluate_trade(ticker,close_s,vol_s,vix_s,spy_s,sect_s,t10,t5)
            results.append(r)
            print(f"  {ticker:<6} DMS:{r['DMS']:>5}  "
                  f"WinP:{r['Win Prob']:>6}  Z:{r['Z-Score']:>5}  "
                  f"Kelly:{r['Kelly %']:>5}  EV:${r['EV $']:>6}  {r['TRADE']}")
        except Exception as e:
            print(f"  {ticker}: error — {e}")

    df = pd.DataFrame(results)
    trades = df[df["TRADE"]=="✅ TRADE"]

    print(f"\n{'='*70}")
    print(f"  TRADE TODAY ({len(trades)}): {', '.join(trades['Ticker'].tolist()) or 'None'}")
    print(f"  All 5 layers must pass for a trade to be placed.")
    print(f"{'='*70}\n")
    return df

if __name__ == "__main__":
    run_model()
