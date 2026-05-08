"""
Equity Model — Full Watchlist Backtester
Runs the 5-factor model across every stock and ranks by performance.
Run with: python backtest.py
Dependencies: pip install yfinance pandas numpy matplotlib
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
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

START        = "2022-01-01"
END          = "2024-12-31"
CAPITAL      = 10000
POSITION_PCT = 0.10
ATR_MULT     = 1.5

WEIGHTS = {
    "price_vol":   0.30,
    "sector_rs":   0.25,
    "breadth":     0.20,
    "volatility":  0.15,
    "yield_curve": 0.10,
}

# ── DATA FETCH ────────────────────────────────────────────────────────────────

def fetch_data():
    all_tix = list(set(
        WATCHLIST + list(SECTOR_MAP.values()) + ["SPY","^VIX","^TNX","^FVX"]
    ))
    print(f"Fetching data for {len(WATCHLIST)} stocks ({START} → {END})...")
    raw = yf.download(all_tix, start=START, end=END,
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
    try:
        c  = close.iloc[:i+1]
        v  = volume.iloc[:i+1]
        vx = vix.iloc[:i+1]
        sp = spy.iloc[:i+1]
        sc = sect.iloc[:i+1]
        if len(c) < 25: return None

        avg_vol   = v.rolling(20).mean().iloc[-1]
        vol_ratio = v.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
        pct_chg   = (c.iloc[-1] - c.iloc[-2]) / c.iloc[-2]
        pv        = normalise(pct_chg * vol_ratio, -0.05, 0.05)

        vix_score = 100 - normalise(vx.iloc[-1], 10, 40)
        hi_c = c.rolling(2).max(); lo_c = c.rolling(2).min()
        atr_pct = ((hi_c - lo_c) / c).rolling(14).mean().iloc[-1] * 100
        vlt = (vix_score + (100 - normalise(atr_pct, 0.5, 5.0))) / 2

        ma50 = sp.rolling(50).mean().iloc[-1]
        brd  = normalise(sp.iloc[-1] / ma50, 0.90, 1.10)

        try:
            slope = t10.iloc[:i+1].iloc[-1] - t5.iloc[:i+1].iloc[-1]
            yld   = normalise(slope, -0.5, 2.0)
        except Exception:
            yld = 50.0

        w  = min(20, len(c)-1)
        sr = (c.iloc[-1] / c.iloc[-w]) - 1
        er = (sc.iloc[-1] / sc.iloc[-w]) - 1
        rs = (1 + sr) / (1 + er) if (1 + er) != 0 else 1.0
        rs_score = normalise(rs, 0.70, 1.30)

        return round(
            WEIGHTS["price_vol"]   * pv       +
            WEIGHTS["sector_rs"]   * rs_score +
            WEIGHTS["breadth"]     * brd      +
            WEIGHTS["volatility"]  * vlt      +
            WEIGHTS["yield_curve"] * yld, 2
        )
    except Exception:
        return None

# ── SINGLE STOCK BACKTEST ─────────────────────────────────────────────────────

def backtest_ticker(ticker, raw):
    close  = get_col(raw, ticker,                   "Close")
    volume = get_col(raw, ticker,                   "Volume")
    vix    = get_col(raw, "^VIX",                   "Close")
    spy    = get_col(raw, "SPY",                    "Close")
    sect   = get_col(raw, SECTOR_MAP.get(ticker,"SPY"), "Close")
    t10    = get_col(raw, "^TNX",                   "Close")
    t5     = get_col(raw, "^FVX",                   "Close")

    if len(close) < 60: return None, None, None

    idx   = close.index
    vix   = vix.reindex(idx,  method="ffill")
    spy   = spy.reindex(idx,  method="ffill")
    sect  = sect.reindex(idx, method="ffill")
    t10   = t10.reindex(idx,  method="ffill")
    t5    = t5.reindex(idx,   method="ffill")
    atr_s = compute_atr(close)

    capital  = float(CAPITAL)
    position = 0
    entry_px = 0.0
    stop_px  = 0.0
    trades   = []
    equity   = []

    for i in range(len(close)):
        price = close.iloc[i]
        date  = idx[i]
        atr   = atr_s.iloc[i] if not pd.isna(atr_s.iloc[i]) else 0
        dms   = compute_dms(i, close, volume, vix, spy, sect, t10, t5)

        if position > 0 and price <= stop_px:
            pnl      = (price - entry_px) * position
            capital += position * price
            trades.append({"Date":date,"Action":"STOP","Price":round(price,2),
                            "Shares":position,"PnL":round(pnl,2)})
            position = 0

        if dms is not None:
            if dms >= 60 and position == 0 and vix.iloc[i] < 30:
                shares = int(capital * POSITION_PCT / price)
                if shares > 0:
                    capital -= shares * price
                    position = shares
                    entry_px = price
                    stop_px  = price - ATR_MULT * atr
                    trades.append({"Date":date,"Action":"BUY","Price":round(price,2),
                                   "Shares":shares,"PnL":0})
            elif dms <= 40 and position > 0:
                pnl      = (price - entry_px) * position
                capital += position * price
                trades.append({"Date":date,"Action":"SELL","Price":round(price,2),
                                "Shares":position,"PnL":round(pnl,2)})
                position = 0

        equity.append(capital + position * price)

    if position > 0:
        price    = close.iloc[-1]
        pnl      = (price - entry_px) * position
        capital += position * price
        trades.append({"Date":idx[-1],"Action":"CLOSE","Price":round(price,2),
                        "Shares":position,"PnL":round(pnl,2)})

    equity_s  = pd.Series(equity, index=idx)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, equity_s, close

# ── METRICS ───────────────────────────────────────────────────────────────────

def calc_metrics(ticker, trades_df, equity_s, close):
    total_ret = (equity_s.iloc[-1] - CAPITAL) / CAPITAL * 100
    bh_ret    = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
    alpha     = total_ret - bh_ret

    daily_ret = equity_s.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                 if daily_ret.std() > 0 else 0)

    roll_max  = equity_s.cummax()
    max_dd    = ((equity_s - roll_max) / roll_max * 100).min()

    closed    = trades_df[trades_df["Action"].isin(["SELL","STOP","CLOSE"])] \
                if not trades_df.empty else pd.DataFrame()
    wins      = closed[closed["PnL"] > 0]  if not closed.empty else pd.DataFrame()
    losses    = closed[closed["PnL"] <= 0] if not closed.empty else pd.DataFrame()
    n_trades  = len(closed)
    hit_rate  = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win   = wins["PnL"].mean()   if len(wins)   > 0 else 0
    avg_loss  = losses["PnL"].mean() if len(losses) > 0 else 0
    pf        = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        "Ticker":          ticker,
        "Total Return %":  round(total_ret, 1),
        "Buy&Hold %":      round(bh_ret, 1),
        "Alpha %":         round(alpha, 1),
        "Sharpe":          round(sharpe, 2),
        "Max DD %":        round(max_dd, 1),
        "Trades":          n_trades,
        "Win Rate %":      round(hit_rate, 1),
        "Profit Factor":   round(pf, 2),
        "Final $":         round(equity_s.iloc[-1], 0),
    }

# ── SUMMARY CHARTS ────────────────────────────────────────────────────────────

def plot_summary(results_df, all_equity):
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#0e1117")

    def style(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor("#1c2030")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # 1 — Ranked bar: Total Return vs Buy & Hold
    ax1 = fig.add_subplot(3, 2, 1)
    rd  = results_df.sort_values("Total Return %", ascending=True)
    x   = np.arange(len(rd))
    ax1.barh(x - 0.2, rd["Total Return %"], height=0.4,
             color="#00e676", label="Model")
    ax1.barh(x + 0.2, rd["Buy&Hold %"],    height=0.4,
             color="#90caf9", label="Buy & Hold")
    ax1.set_yticks(x); ax1.set_yticklabels(rd["Ticker"], color="white")
    ax1.axvline(0, color="#555", linewidth=0.8)
    ax1.legend(facecolor="#1c2030", labelcolor="white", fontsize=8)
    style(ax1, "Total Return % — Model vs Buy & Hold", "Return %")

    # 2 — Sharpe ratio ranked
    ax2  = fig.add_subplot(3, 2, 2)
    rd2  = results_df.sort_values("Sharpe", ascending=True)
    cols = ["#00e676" if v >= 1.0 else "#ffd740" if v >= 0 else "#ff1744"
            for v in rd2["Sharpe"]]
    ax2.barh(rd2["Ticker"], rd2["Sharpe"], color=cols)
    ax2.axvline(1.0, color="#ffd740", linestyle="--", linewidth=1,
                label="Sharpe = 1.0")
    ax2.legend(facecolor="#1c2030", labelcolor="white", fontsize=8)
    style(ax2, "Sharpe Ratio Ranked", "Sharpe")

    # 3 — Alpha (outperformance vs buy & hold)
    ax3  = fig.add_subplot(3, 2, 3)
    rd3  = results_df.sort_values("Alpha %", ascending=True)
    cols = ["#00e676" if v > 0 else "#ff1744" for v in rd3["Alpha %"]]
    ax3.barh(rd3["Ticker"], rd3["Alpha %"], color=cols)
    ax3.axvline(0, color="#555", linewidth=0.8)
    style(ax3, "Alpha vs Buy & Hold (%)", "Alpha %")

    # 4 — Win rate vs profit factor scatter
    ax4 = fig.add_subplot(3, 2, 4)
    sc  = ax4.scatter(results_df["Win Rate %"], results_df["Profit Factor"],
                      c=results_df["Sharpe"], cmap="RdYlGn",
                      s=120, zorder=5, vmin=0, vmax=2)
    for _, r in results_df.iterrows():
        ax4.annotate(r["Ticker"], (r["Win Rate %"], r["Profit Factor"]),
                     fontsize=8, color="white",
                     xytext=(4, 4), textcoords="offset points")
    ax4.axvline(50, color="#555", linestyle="--", linewidth=0.8)
    ax4.axhline(1,  color="#555", linestyle="--", linewidth=0.8)
    plt.colorbar(sc, ax=ax4, label="Sharpe").ax.yaxis.label.set_color("white")
    style(ax4, "Win Rate vs Profit Factor", "Win Rate %", "Profit Factor")

    # 5 — All equity curves
    ax5 = fig.add_subplot(3, 1, 3)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_equity)))
    for (ticker, eq), col in zip(all_equity.items(), colors):
        ax5.plot(eq.index, eq.values, linewidth=1.5, label=ticker, color=col)
    ax5.axhline(CAPITAL, color="#555", linestyle=":", linewidth=1)
    ax5.legend(facecolor="#1c2030", labelcolor="white", fontsize=8,
               ncol=5, loc="upper left")
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    style(ax5, "Equity Curves — All Stocks", "Date", "Portfolio Value ($)")

    plt.tight_layout(pad=2.5)
    fname = f"backtest_watchlist_{START[:4]}_{END[:4]}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    print(f"\n  Chart saved → {fname}")
    plt.show()

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw = fetch_data()

    all_results = []
    all_equity  = {}

    for ticker in WATCHLIST:
        print(f"  Backtesting {ticker}...", end=" ")
        trades_df, equity_s, close = backtest_ticker(ticker, raw)
        if equity_s is None:
            print("skipped (insufficient data)")
            continue
        m = calc_metrics(ticker, trades_df, equity_s, close)
        all_results.append(m)
        all_equity[ticker] = equity_s
        print(f"Return: {m['Total Return %']:+.1f}%  |  Sharpe: {m['Sharpe']}  |  Alpha: {m['Alpha %']:+.1f}%")

    results_df = pd.DataFrame(all_results).sort_values("Sharpe", ascending=False)

    # ── Print leaderboard ──
    print(f"\n{'='*90}")
    print(f"  WATCHLIST BACKTEST LEADERBOARD — {START} → {END}")
    print(f"{'='*90}")
    print(results_df.to_string(index=False))
    print(f"{'='*90}")

    # ── Best / worst ──
    best  = results_df.iloc[0]
    worst = results_df.iloc[-1]
    print(f"\n🏆 Best  model fit: {best['Ticker']}  "
          f"(Sharpe {best['Sharpe']}, Return {best['Total Return %']:+.1f}%, "
          f"Alpha {best['Alpha %']:+.1f}%)")
    print(f"⚠️  Worst model fit: {worst['Ticker']}  "
          f"(Sharpe {worst['Sharpe']}, Return {worst['Total Return %']:+.1f}%, "
          f"Alpha {worst['Alpha %']:+.1f}%)")

    # ── Save CSV ──
    fname = f"backtest_results_{START[:4]}_{END[:4]}.csv"
    results_df.to_csv(fname, index=False)
    print(f"\n  Results saved → {fname}")

    plot_summary(results_df, all_equity)
