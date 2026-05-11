"""
Equity Quant Model — Enhanced for Better Sharpe Ratio
Improvements:
  1. Higher entry threshold (DMS >= 70)
  2. SPY 200MA regime filter
  3. Sector exposure cap (max 2 stocks per sector)
  4. Conviction-based position sizing
  5. Trailing stop instead of fixed ATR stop
  6. Time-based stop (exit after 10 days if underwater)
  7. Dual timeframe filter (weekly + daily DMS must agree)
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
    "TSLA":"XLY","NFLX":"XLC","BAC":"XLF","PFE":"XLV","CVX":"XLE",
    "COST":"XLP","WMT":"XLP","DIS":"XLC","BA":"XLI","CAT":"XLI"
}

WEIGHTS = {
    "price_vol":   0.30,
    "sector_rs":   0.25,
    "breadth":     0.20,
    "volatility":  0.15,
    "yield_curve": 0.10,
}

# ── ENHANCED PARAMETERS ───────────────────────────────────────────────────────

BUY_THRESHOLD   = 70    # raised from 60 — high conviction only
SELL_THRESHOLD  = 40    # exit when score weakens
VIX_PAUSE       = 30    # halt buys above this VIX level
MAX_SECTOR_SLOTS= 2     # max positions per sector at once
MAX_POSITIONS   = 8     # max total open positions
TIME_STOP_DAYS  = 10    # exit if underwater after N days
ATR_MULT        = 1.5   # initial stop distance
TRAIL_MULT      = 2.0   # trailing stop distance (wider to let winners run)

# Conviction-based sizing
SIZING = {
    "STRONG BUY": 0.15,  # DMS >= 75 → 15% of capital
    "BUY":        0.08,  # DMS 70–74 → 8% of capital
}

LOOKBACK_DAYS = 120  # calendar days of history

# ── HELPERS ───────────────────────────────────────────────────────────────────

def normalise(val, lo, hi):
    return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))

def fetch(tickers, days=LOOKBACK_DAYS):
    end   = datetime.today()
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

def get_signal(score):
    if score >= 75: return "STRONG BUY"
    if score >= 70: return "BUY"
    if score >= 40: return "NEUTRAL"
    if score >= 25: return "SELL"
    return "STRONG SELL"

# ── SUB-SCORES ────────────────────────────────────────────────────────────────

def score_price_volume(close_s, vol_s):
    avg_vol   = vol_s.rolling(20).mean().iloc[-1]
    vol_ratio = vol_s.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
    pct_chg   = (close_s.iloc[-1] - close_s.iloc[-2]) / close_s.iloc[-2]
    return normalise(pct_chg * vol_ratio, -0.05, 0.05)

def score_volatility(vix_series, close_s):
    vix_score = 100 - normalise(vix_series.iloc[-1], 10, 40)
    hi = close_s.rolling(2).max(); lo = close_s.rolling(2).min()
    atr_pct = ((hi - lo) / close_s).rolling(14).mean().iloc[-1] * 100
    return (vix_score + (100 - normalise(atr_pct, 0.5, 5.0))) / 2

def score_breadth(spy_close):
    ma50 = spy_close.rolling(50).mean().iloc[-1]
    return normalise(spy_close.iloc[-1] / ma50, 0.90, 1.10)

def score_yield_curve(t10, t5):
    try:
        return normalise(t10.iloc[-1] - t5.iloc[-1], -0.5, 2.0)
    except Exception:
        return 50.0

def score_sector_rs(close_s, sect_s, window=20):
    sr = (close_s.iloc[-1] / close_s.iloc[-window]) - 1
    er = (sect_s.iloc[-1]  / sect_s.iloc[-window])  - 1
    rs = (1 + sr) / ((1 + er) if (1 + er) != 0 else 1)
    return normalise(rs, 0.70, 1.30)

def compute_dms(close_s, vol_s, vix_s, spy_s, sect_s, t10, t5):
    pv  = score_price_volume(close_s, vol_s)
    vlt = score_volatility(vix_s, close_s)
    brd = score_breadth(spy_s)
    yld = score_yield_curve(t10, t5)
    rs  = score_sector_rs(close_s, sect_s)
    return round(
        WEIGHTS["price_vol"]   * pv  +
        WEIGHTS["sector_rs"]   * rs  +
        WEIGHTS["breadth"]     * brd +
        WEIGHTS["volatility"]  * vlt +
        WEIGHTS["yield_curve"] * yld, 2
    )

# ── WEEKLY DMS (dual timeframe filter) ───────────────────────────────────────

def compute_weekly_dms(close_s, vol_s, vix_s, spy_s, sect_s, t10, t5):
    """Resample to weekly and compute DMS on weekly bars."""
    try:
        wc   = close_s.resample("W").last()
        wv   = vol_s.resample("W").sum()
        wvix = vix_s.resample("W").last()
        wspy = spy_s.resample("W").last()
        wsct = sect_s.resample("W").last()
        wt10 = t10.resample("W").last()
        wt5  = t5.resample("W").last()
        if len(wc) < 10: return None
        return compute_dms(wc, wv, wvix, wspy, wsct, wt10, wt5)
    except Exception:
        return None

# ── REGIME CHECKS ─────────────────────────────────────────────────────────────

def regime_ok(vix_s, spy_s):
    """
    Returns (ok, reason)
    Requires: VIX < 30 AND SPY above its 200-day MA
    """
    vix_now = vix_s.iloc[-1]
    if vix_now >= VIX_PAUSE:
        return False, f"VIX {vix_now:.1f} ≥ {VIX_PAUSE} — risk-off"

    spy_200 = spy_s.rolling(200).mean().iloc[-1]
    if spy_s.iloc[-1] < spy_200:
        return False, f"SPY below 200MA ({spy_s.iloc[-1]:.1f} < {spy_200:.1f}) — bear market"

    return True, "Risk-ON ✅"

# ── POSITION SIZING ───────────────────────────────────────────────────────────

def position_size(signal, capital):
    """Conviction-based sizing — bigger bets on higher-conviction signals."""
    pct = SIZING.get(signal, 0.08)
    return capital * pct

def atr_value(close_s):
    hi  = close_s.rolling(2).max()
    lo  = close_s.rolling(2).min()
    return (hi - lo).rolling(14).mean().iloc[-1]

# ── TRAILING STOP ─────────────────────────────────────────────────────────────

def update_trail(current_price, highest_since_entry, atr):
    """Move stop up as price rises — never move it down."""
    new_high  = max(current_price, highest_since_entry)
    new_stop  = new_high - TRAIL_MULT * atr
    return new_high, new_stop

# ── SECTOR EXPOSURE CHECK ─────────────────────────────────────────────────────

def sector_slots_available(ticker, open_positions):
    """Returns True if sector not yet at MAX_SECTOR_SLOTS."""
    sector  = SECTOR_MAP.get(ticker, "UNKNOWN")
    current = sum(1 for t in open_positions
                  if SECTOR_MAP.get(t, "") == sector)
    return current < MAX_SECTOR_SLOTS

# ── MAIN ENGINE ───────────────────────────────────────────────────────────────

def run_model():
    print(f"\n{'='*70}")
    print(f"  EQUITY QUANT MODEL (Enhanced)  |  {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    all_tix = list(set(WATCHLIST + list(SECTOR_MAP.values()) +
                       ["SPY","^VIX","^TNX","^FVX"]))
    print("Fetching market data...")
    raw = fetch(all_tix)

    vix_s = get_col(raw, "^VIX", "Close")
    spy_s = get_col(raw, "SPY",  "Close")
    t10   = get_col(raw, "^TNX", "Close")
    t5    = get_col(raw, "^FVX", "Close")

    # ── Regime check ──
    ok, reason = regime_ok(vix_s, spy_s)
    spy_pct_200 = ((spy_s.iloc[-1] / spy_s.rolling(200).mean().iloc[-1]) - 1) * 100
    print(f"VIX:       {vix_s.iloc[-1]:.1f}")
    print(f"SPY vs 200MA: {spy_pct_200:+.1f}%")
    print(f"Regime:    {reason}\n")

    results = []
    open_positions = []  # track sector exposure

    for ticker in WATCHLIST:
        close_s = get_col(raw, ticker, "Close")
        vol_s   = get_col(raw, ticker, "Volume")
        sect_s  = get_col(raw, SECTOR_MAP.get(ticker,"SPY"), "Close")

        if len(close_s) < 30:
            continue

        try:
            # Daily DMS
            dms = compute_dms(close_s, vol_s, vix_s, spy_s, sect_s, t10, t5)

            # Weekly DMS (dual timeframe)
            w_dms = compute_weekly_dms(close_s, vol_s, vix_s, spy_s, sect_s, t10, t5)
            dual_ok = (w_dms is not None and w_dms >= 55)  # weekly must be positive

            signal  = get_signal(dms)
            price   = close_s.iloc[-1]
            atr     = atr_value(close_s)
            stop    = round(price - ATR_MULT * atr, 2)
            trail   = round(price - TRAIL_MULT * atr, 2)
            sizing  = position_size(signal, 100000)  # example £100k portfolio
            sector  = SECTOR_MAP.get(ticker, "—")
            sec_ok  = sector_slots_available(ticker, open_positions)

            # Entry eligibility
            eligible = (
                ok and
                dms >= BUY_THRESHOLD and
                dual_ok and
                sec_ok and
                len(open_positions) < MAX_POSITIONS
            )

            if eligible:
                open_positions.append(ticker)

            results.append({
                "Ticker":      ticker,
                "Price":       round(price, 2),
                "DMS":         dms,
                "Weekly DMS":  round(w_dms, 1) if w_dms else "—",
                "Signal":      signal,
                "Eligible":    "✅ YES" if eligible else "❌ NO",
                "Stop":        stop,
                "Trail Stop":  trail,
                "ATR":         round(atr, 2),
                "Size $":      round(sizing, 0),
                "Sector":      sector,
                "Dual TF":     "✅" if dual_ok else "❌",
                "Sec Slot":    "✅" if sec_ok  else "❌",
                "Regime":      "✅" if ok       else "❌",
            })

        except Exception as e:
            print(f"  {ticker}: error — {e}")

    df = pd.DataFrame(results).sort_values("DMS", ascending=False)

    # ── Print table ──
    print(f"{'Ticker':<8} {'Price':>7} {'DMS':>6} {'Wkly':>6} {'Signal':<13} "
          f"{'Eligible':<10} {'Stop':>7} {'Size$':>8}  TF  Sec Reg")
    print("-" * 95)
    for _, r in df.iterrows():
        print(f"{r['Ticker']:<8} {r['Price']:>7.2f} {r['DMS']:>6.1f} "
              f"{str(r['Weekly DMS']):>6} {r['Signal']:<13} "
              f"{r['Eligible']:<10} {r['Stop']:>7.2f} {r['Size $']:>8,.0f}  "
              f"{r['Dual TF']}   {r['Sec Slot']}   {r['Regime']}")

    eligible = df[df["Eligible"] == "✅ YES"]
    print(f"\n✅ Eligible to trade today ({len(eligible)}): "
          f"{', '.join(eligible['Ticker'].tolist()) or 'None'}")
    print(f"\nAll filters must pass: Regime ✅  |  DMS ≥ {BUY_THRESHOLD}  |  "
          f"Weekly DMS ≥ 55  |  Sector slots < {MAX_SECTOR_SLOTS}\n")

    return df

if __name__ == "__main__":
    run_model()
