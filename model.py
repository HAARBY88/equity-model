"""
Equity Quantitative Scoring Model
Combines 5 market metrics into a Daily Market Score (DMS) per stock.
Dependencies: pip install yfinance pandas numpy requests
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

WATCHLIST = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "JPM", "XOM", "UNH", "V"]

SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOGL": "XLC", "META": "XLC",
    "AMZN": "XLY", "JPM": "XLF", "XOM": "XLE", "UNH": "XLV", "V": "XLF"
}

WEIGHTS = {
    "price_vol":  0.30,
    "sector_rs":  0.25,
    "breadth":    0.20,
    "volatility": 0.15,
    "yield_curve":0.10,
}

SIGNAL_THRESHOLDS = {
    (75, 100): ("STRONG BUY",  "Full position"),
    (60,  74): ("BUY",         "Half position"),
    (40,  59): ("NEUTRAL",     "Hold / watch"),
    (25,  39): ("SELL",        "Reduce position"),
    (0,   24): ("STRONG SELL", "Exit / short"),
}

VIX_PAUSE_THRESHOLD = 30  # halt all buys above this level
LOOKBACK_DAYS       = 60  # calendar days of history to fetch


# ── HELPERS ───────────────────────────────────────────────────────────────────

def normalise(val, lo, hi):
    """Clip and scale a value to 0–100."""
    return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))


def fetch(tickers, days=LOOKBACK_DAYS):
    end   = datetime.today()
    start = end - timedelta(days=days)
    raw   = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    return raw


def get_signal(score):
    for (lo, hi), (label, action) in SIGNAL_THRESHOLDS.items():
        if lo <= score <= hi:
            return label, action
    return "NEUTRAL", "Hold / watch"


# ── SUB-SCORES ────────────────────────────────────────────────────────────────

def score_price_volume(close_s, vol_s):
    """30% weight — momentum confirmed by volume."""
    avg_vol   = vol_s.rolling(20).mean().iloc[-1]
    vol_ratio = vol_s.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
    pct_chg   = (close_s.iloc[-1] - close_s.iloc[-2]) / close_s.iloc[-2]
    raw       = pct_chg * vol_ratio          # positive = bullish
    return normalise(raw, -0.05, 0.05)       # ±5% daily range


def score_volatility(vix_series, close_s):
    """15% weight — low vol = risk-on (inverted)."""
    vix_latest = vix_series.iloc[-1]
    vix_score  = 100 - normalise(vix_latest, 10, 40)

    # ATR (14-day)
    hi = close_s.rolling(2).max()
    lo = close_s.rolling(2).min()
    atr_pct = ((hi - lo) / close_s).rolling(14).mean().iloc[-1] * 100
    atr_score = 100 - normalise(atr_pct, 0.5, 5.0)

    return (vix_score + atr_score) / 2


def score_breadth(spy_data):
    """20% weight — % of SPY constituents above their 50-day MA.
    Approximated here using SPY itself vs its 50MA."""
    ma50  = spy_data["Close"].rolling(50).mean().iloc[-1]
    price = spy_data["Close"].iloc[-1]
    # Proxy: 100 if price > MA50, graded by distance
    ratio = price / ma50
    return normalise(ratio, 0.90, 1.10)


def score_yield_curve(treasury_data):
    """10% weight — steeper curve = risk-on."""
    # Uses ^TNX (10y) and ^FVX (5y) as proxy (^IRX = 13-week)
    if treasury_data is None:
        return 50.0  # neutral fallback
    try:
        t10 = treasury_data["^TNX"]["Close"].iloc[-1]
        t2  = treasury_data["^FVX"]["Close"].iloc[-1]
        slope = t10 - t2
        return normalise(slope, -0.5, 2.0)
    except Exception:
        return 50.0


def score_sector_rs(close_s, sector_close_s, window=20):
    """25% weight — stock return vs sector ETF over last N days."""
    stock_ret  = (close_s.iloc[-1]  / close_s.iloc[-window])  - 1
    sector_ret = (sector_close_s.iloc[-1] / sector_close_s.iloc[-window]) - 1
    rs = (1 + stock_ret) / (1 + sector_ret) if (1 + sector_ret) != 0 else 1.0
    return normalise(rs, 0.70, 1.30)


# ── POSITION SIZING ───────────────────────────────────────────────────────────

def kelly_size(win_rate=0.55, avg_win=0.03, avg_loss=0.015, max_pct=0.10):
    """Fractional Kelly (half-Kelly) for position sizing."""
    b = avg_win / avg_loss
    k = (b * win_rate - (1 - win_rate)) / b
    return round(min(k / 2, max_pct) * 100, 1)  # return as %


def atr_stop(close_s, multiplier=1.5):
    """Stop-loss level = entry - 1.5 × ATR(14)."""
    hi = close_s.rolling(2).max()
    lo = close_s.rolling(2).min()
    atr = ((hi - lo)).rolling(14).mean().iloc[-1]
    entry = close_s.iloc[-1]
    stop  = entry - multiplier * atr
    return round(stop, 2), round(atr, 2)


# ── MAIN ENGINE ───────────────────────────────────────────────────────────────

def run_model():
    print(f"\n{'='*65}")
    print(f"  EQUITY QUANT MODEL  |  {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}\n")

    # ── Fetch all data ──
    all_tickers = list(set(WATCHLIST + list(SECTOR_MAP.values()) +
                           ["SPY", "^VIX", "^TNX", "^FVX"]))
    print("Fetching market data...")
    raw = fetch(all_tickers)

    # Normalise column access (single vs multi-ticker)
    def get_close(ticker):
        try:
            return raw["Close"][ticker].dropna()
        except KeyError:
            return pd.Series(dtype=float)

    def get_vol(ticker):
        try:
            return raw["Volume"][ticker].dropna()
        except KeyError:
            return pd.Series(dtype=float)

    vix_series = get_close("^VIX")
    spy_data   = {"Close": get_close("SPY")}

    # Treasury proxy bundle
    treasury_data = {"^TNX": {"Close": get_close("^TNX")},
                     "^FVX": {"Close": get_close("^FVX")}}

    # ── Regime filter ──
    vix_now = vix_series.iloc[-1] if not vix_series.empty else 20
    regime_ok = vix_now < VIX_PAUSE_THRESHOLD
    print(f"VIX: {vix_now:.1f}  |  Regime: {'✅ Risk-ON' if regime_ok else '🚨 Risk-OFF — buys paused'}\n")

    # ── Score each stock ──
    results = []

    for ticker in WATCHLIST:
        close_s = get_close(ticker)
        vol_s   = get_vol(ticker)
        sector  = SECTOR_MAP.get(ticker, "SPY")
        sect_s  = get_close(sector)

        if len(close_s) < 25:
            print(f"  {ticker}: insufficient data, skipping.")
            continue

        try:
            pv   = score_price_volume(close_s, vol_s)
            vlt  = score_volatility(vix_series, close_s)
            brd  = score_breadth(spy_data)
            yld  = score_yield_curve(treasury_data)
            rs   = score_sector_rs(close_s, sect_s)

            dms = (WEIGHTS["price_vol"]   * pv  +
                   WEIGHTS["sector_rs"]   * rs  +
                   WEIGHTS["breadth"]     * brd +
                   WEIGHTS["volatility"]  * vlt +
                   WEIGHTS["yield_curve"] * yld)

            signal, action = get_signal(dms)
            stop, atr_val  = atr_stop(close_s)
            kelly          = kelly_size()

            results.append({
                "Ticker":    ticker,
                "Price":     round(close_s.iloc[-1], 2),
                "DMS":       round(dms, 1),
                "Signal":    signal,
                "Action":    action,
                "Stop":      stop,
                "ATR":       round(atr_val, 2),
                "Kelly%":    kelly,
                "PV":        round(pv, 1),
                "VolScore":  round(vlt, 1),
                "Breadth":   round(brd, 1),
                "Yield":     round(yld, 1),
                "SectorRS":  round(rs, 1),
            })

        except Exception as e:
            print(f"  {ticker}: error — {e}")

    if not results:
        print("No results generated.")
        return

    # ── Output table ──
    df = pd.DataFrame(results).sort_values("DMS", ascending=False)

    print(f"{'Ticker':<8} {'Price':>7} {'DMS':>6} {'Signal':<13} {'Stop':>7} {'ATR':>6} {'Kelly%':>7}  Sub-scores (PV/Vol/Brd/Yld/RS)")
    print("-" * 85)
    for _, r in df.iterrows():
        buy_flag = "🔒" if not regime_ok and "BUY" in r["Signal"] else ""
        print(f"{r['Ticker']:<8} {r['Price']:>7.2f} {r['DMS']:>6.1f} {r['Signal']:<13} "
              f"{r['Stop']:>7.2f} {r['ATR']:>6.2f} {r['Kelly%']:>6.1f}%  "
              f"{r['PV']:.0f} / {r['VolScore']:.0f} / {r['Breadth']:.0f} / {r['Yield']:.0f} / {r['SectorRS']:.0f}  {buy_flag}")

    # ── Top picks ──
    buys = df[df["Signal"].isin(["STRONG BUY", "BUY"])]
    sells = df[df["Signal"].isin(["STRONG SELL", "SELL"])]

    print(f"\n📈 BUY candidates  ({len(buys)}): {', '.join(buys['Ticker'].tolist()) or 'None'}")
    print(f"📉 SELL candidates ({len(sells)}): {', '.join(sells['Ticker'].tolist()) or 'None'}")
    print(f"\nMax position size (Kelly): {kelly_size()}% of portfolio per stock")
    print(f"Stop-loss rule: Entry − 1.5 × ATR(14)\n")

    return df


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results_df = run_model()
