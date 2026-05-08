"""
Equity Model Backtester — backtest.py
Tests the 5-factor scoring model on a single stock over a date range.
Run with: python backtest.py
Dependencies: pip install yfinance pandas numpy matplotlib
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKER      = "AAPL"       # Stock to backtest
SECTOR_ETF  = "XLK"        # Corresponding sector ETF
START       = "2022-01-01" # Backtest start date
END         = "2024-12-31" # Backtest end date
CAPITAL     = 10000        # Starting capital £/$ 
POSITION_PCT= 0.10         # 10% of capital per trade (fixed fractional)
ATR_MULT    = 1.5          # Stop-loss multiplier

WEIGHTS = {
    "price_vol":   0.30,
    "sector_rs":   0.25,
    "breadth":     0.20,
    "volatility":  0.15,
    "yield_curve": 0.10,
}

# ── DATA FETCH ────────────────────────────────────────────────────────────────

def fetch_data():
    print(f"Fetching data for {TICKER} ({START} → {END})...")
    tickers = [TICKER, SECTOR_ETF, "SPY", "^VIX", "^TNX", "^FVX"]
    raw = yf.download(tickers, start=START, end=END,
                      auto_adjust=True, progress=False, group_by="ticker")
    print(f"  {len(raw)} trading days loaded.\n")
    return raw

def get_col(raw, ticker, field):
    try:
        if (ticker, field) in raw.columns:
            return raw[(ticker, field)].dropna()
        if field in raw.columns:
            return raw[field].dropna()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

# ── INDICATORS ────────────────────────────────────────────────────────────────

def normalise(val, lo, hi):
    return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))

def compute_atr(close, window=14):
    hi = close.rolling(2).max()
    lo = close.rolling(2).min()
    return (hi - lo).rolling(window).mean()

def compute_dms(i, close, volume, vix, spy, sect, t10, t5):
    """Compute DMS for a single row index i using only data up to i."""
    try:
        c  = close.iloc[:i+1]
        v  = volume.iloc[:i+1]
        vx = vix.iloc[:i+1]
        sp = spy.iloc[:i+1]
        sc = sect.iloc[:i+1]

        if len(c) < 25: return None

        # Price / Volume
        avg_vol   = v.rolling(20).mean().iloc[-1]
        vol_ratio = v.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
        pct_chg   = (c.iloc[-1] - c.iloc[-2]) / c.iloc[-2]
        pv        = normalise(pct_chg * vol_ratio, -0.05, 0.05)

        # Volatility
        vix_score = 100 - normalise(vx.iloc[-1], 10, 40)
        hi_c = c.rolling(2).max(); lo_c = c.rolling(2).min()
        atr_pct = ((hi_c - lo_c) / c).rolling(14).mean().iloc[-1] * 100
        vlt = (vix_score + (100 - normalise(atr_pct, 0.5, 5.0))) / 2

        # Breadth
        ma50 = sp.rolling(50).mean().iloc[-1]
        brd  = normalise(sp.iloc[-1] / ma50, 0.90, 1.10)

        # Yield curve
        try:
            slope = t10.iloc[:i+1].iloc[-1] - t5.iloc[:i+1].iloc[-1]
            yld   = normalise(slope, -0.5, 2.0)
        except Exception:
            yld = 50.0

        # Sector RS
        w = min(20, len(c)-1)
        sr = (c.iloc[-1] / c.iloc[-w]) - 1
        er = (sc.iloc[-1] / sc.iloc[-w]) - 1
        rs = (1 + sr) / (1 + er) if (1 + er) != 0 else 1.0
        rs_score = normalise(rs, 0.70, 1.30)

        dms = (WEIGHTS["price_vol"]   * pv       +
               WEIGHTS["sector_rs"]   * rs_score +
               WEIGHTS["breadth"]     * brd      +
               WEIGHTS["volatility"]  * vlt      +
               WEIGHTS["yield_curve"] * yld)

        return round(dms, 2)

    except Exception:
        return None

# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────

def run_backtest(raw):
    close  = get_col(raw, TICKER,     "Close")
    volume = get_col(raw, TICKER,     "Volume")
    vix    = get_col(raw, "^VIX",     "Close")
    spy    = get_col(raw, "SPY",      "Close")
    sect   = get_col(raw, SECTOR_ETF, "Close")
    t10    = get_col(raw, "^TNX",     "Close")
    t5     = get_col(raw, "^FVX",     "Close")

    # Align all series to stock's trading days
    idx    = close.index
    vix    = vix.reindex(idx,   method="ffill")
    spy    = spy.reindex(idx,   method="ffill")
    sect   = sect.reindex(idx,  method="ffill")
    t10    = t10.reindex(idx,   method="ffill")
    t5     = t5.reindex(idx,    method="ffill")
    atr_s  = compute_atr(close)

    # ── Simulate daily ──
    capital    = float(CAPITAL)
    position   = 0       # shares held
    entry_px   = 0.0
    stop_px    = 0.0
    trades     = []
    equity     = []
    dms_series = []

    for i in range(len(close)):
        price  = close.iloc[i]
        date   = idx[i]
        atr    = atr_s.iloc[i] if not pd.isna(atr_s.iloc[i]) else 0
        dms    = compute_dms(i, close, volume, vix, spy, sect, t10, t5)
        dms_series.append(dms)

        # ── Stop-loss check ──
        if position > 0 and price <= stop_px:
            proceeds = position * price
            pnl      = proceeds - (position * entry_px)
            capital += proceeds
            trades.append({
                "Date": date, "Action": "STOP",
                "Price": round(price,2), "Shares": position,
                "PnL": round(pnl,2), "Capital": round(capital,2)
            })
            position = 0

        if dms is None:
            equity.append(capital + position * price)
            continue

        # ── Signal logic ──
        if dms >= 60 and position == 0 and vix.iloc[i] < 30:
            # BUY
            alloc  = capital * POSITION_PCT
            shares = int(alloc / price)
            if shares > 0:
                cost     = shares * price
                capital -= cost
                position = shares
                entry_px = price
                stop_px  = price - ATR_MULT * atr
                trades.append({
                    "Date": date, "Action": "BUY",
                    "Price": round(price,2), "Shares": shares,
                    "PnL": 0, "Capital": round(capital,2)
                })

        elif dms <= 40 and position > 0:
            # SELL
            proceeds = position * price
            pnl      = proceeds - (position * entry_px)
            capital += proceeds
            trades.append({
                "Date": date, "Action": "SELL",
                "Price": round(price,2), "Shares": position,
                "PnL": round(pnl,2), "Capital": round(capital,2)
            })
            position = 0

        equity.append(capital + position * price)

    # Close any open position at end
    if position > 0:
        price    = close.iloc[-1]
        proceeds = position * price
        pnl      = proceeds - (position * entry_px)
        capital += proceeds
        trades.append({
            "Date": idx[-1], "Action": "CLOSE",
            "Price": round(price,2), "Shares": position,
            "PnL": round(pnl,2), "Capital": round(capital,2)
        })

    trades_df = pd.DataFrame(trades)
    equity_s  = pd.Series(equity, index=idx)
    dms_s     = pd.Series(dms_series, index=idx)

    return trades_df, equity_s, dms_s, close

# ── METRICS ───────────────────────────────────────────────────────────────────

def calc_metrics(trades_df, equity_s, close):
    total_return  = (equity_s.iloc[-1] - CAPITAL) / CAPITAL * 100
    buy_hold      = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

    daily_ret = equity_s.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                 if daily_ret.std() > 0 else 0)

    roll_max  = equity_s.cummax()
    drawdown  = (equity_s - roll_max) / roll_max * 100
    max_dd    = drawdown.min()

    closed = trades_df[trades_df["Action"].isin(["SELL","STOP","CLOSE"])]
    wins   = closed[closed["PnL"] > 0]
    losses = closed[closed["PnL"] <= 0]
    hit_rate   = len(wins) / len(closed) * 100 if len(closed) > 0 else 0
    avg_win    = wins["PnL"].mean()   if len(wins)   > 0 else 0
    avg_loss   = losses["PnL"].mean() if len(losses) > 0 else 0
    profit_fac = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    return {
        "Total Return (%)":    round(total_return, 2),
        "Buy & Hold (%)":      round(buy_hold, 2),
        "Sharpe Ratio":        round(sharpe, 2),
        "Max Drawdown (%)":    round(max_dd, 2),
        "Total Trades":        len(closed),
        "Win Rate (%)":        round(hit_rate, 2),
        "Avg Win ($)":         round(avg_win, 2),
        "Avg Loss ($)":        round(avg_loss, 2),
        "Profit Factor":       round(profit_fac, 2),
        "Final Capital ($)":   round(equity_s.iloc[-1], 2),
    }

# ── CHARTS ────────────────────────────────────────────────────────────────────

def plot_results(trades_df, equity_s, dms_s, close):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={"height_ratios": [3, 2, 1.5]})
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#1c2030")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── Chart 1: Price + trades ──
    axes[0].plot(close.index, close.values, color="#2196F3", linewidth=1.5, label="Price")
    if not trades_df.empty:
        buys  = trades_df[trades_df["Action"] == "BUY"]
        sells = trades_df[trades_df["Action"].isin(["SELL","CLOSE"])]
        stops = trades_df[trades_df["Action"] == "STOP"]
        axes[0].scatter(buys["Date"],  buys["Price"],  marker="^", color="#00e676", s=100, zorder=5, label="Buy")
        axes[0].scatter(sells["Date"], sells["Price"], marker="v", color="#ffd740", s=100, zorder=5, label="Sell")
        axes[0].scatter(stops["Date"], stops["Price"], marker="x", color="#ff1744", s=100, zorder=5, label="Stop")
    axes[0].set_title(f"{TICKER} — Price with Buy/Sell Signals")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend(facecolor="#1c2030", labelcolor="white")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Chart 2: Equity curve vs buy & hold ──
    bh = (close / close.iloc[0]) * CAPITAL
    axes[1].plot(equity_s.index, equity_s.values, color="#00e676", linewidth=1.5, label="Model")
    axes[1].plot(bh.index,       bh.values,       color="#90caf9", linewidth=1.5,
                 linestyle="--", label="Buy & Hold")
    axes[1].axhline(CAPITAL, color="#555", linestyle=":", linewidth=1)
    axes[1].set_title("Equity Curve vs Buy & Hold")
    axes[1].set_ylabel("Portfolio Value ($)")
    axes[1].legend(facecolor="#1c2030", labelcolor="white")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Chart 3: DMS over time ──
    dms_clean = dms_s.dropna()
    axes[2].plot(dms_clean.index, dms_clean.values, color="#ffd740", linewidth=1)
    axes[2].axhline(60, color="#00e676", linestyle="--", linewidth=0.8, label="Buy (60)")
    axes[2].axhline(40, color="#ff6e40", linestyle="--", linewidth=0.8, label="Sell (40)")
    axes[2].fill_between(dms_clean.index, dms_clean.values, 60,
                         where=dms_clean >= 60, alpha=0.2, color="#00e676")
    axes[2].fill_between(dms_clean.index, dms_clean.values, 40,
                         where=dms_clean <= 40, alpha=0.2, color="#ff1744")
    axes[2].set_title("Daily Market Score (DMS)")
    axes[2].set_ylabel("DMS")
    axes[2].set_ylim(0, 100)
    axes[2].legend(facecolor="#1c2030", labelcolor="white")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout(pad=2.0)
    fname = f"backtest_{TICKER}_{START[:4]}_{END[:4]}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    print(f"\n  Chart saved → {fname}")
    plt.show()

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw = fetch_data()
    trades_df, equity_s, dms_s, close = run_backtest(raw)
    metrics = calc_metrics(trades_df, equity_s, close)

    print(f"\n{'='*45}")
    print(f"  BACKTEST RESULTS — {TICKER}  ({START} → {END})")
    print(f"{'='*45}")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print(f"{'='*45}\n")

    print("Trade Log:")
    print(trades_df.to_string(index=False))

    plot_results(trades_df, equity_s, dms_s, close)
