"""
Equity Quant Dashboard — dashboard.py
Run with: streamlit run dashboard.py
Dependencies: pip install yfinance pandas numpy streamlit plotly
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Equity Quant Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .header-bar {
        background: linear-gradient(90deg, #1a237e, #283593);
        padding: 18px 24px; border-radius: 10px; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────

DEFAULT_WATCHLIST = ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","JPM","XOM","UNH","V"]

SECTOR_MAP = {
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","GOOGL":"XLC","META":"XLC",
    "AMZN":"XLY","JPM":"XLF","XOM":"XLE","UNH":"XLV","V":"XLF",
    "TSLA":"XLY","NFLX":"XLC","BAC":"XLF","PFE":"XLV","CVX":"XLE",
    "COST":"XLP","WMT":"XLP","DIS":"XLC","BA":"XLI","CAT":"XLI"
}

SECTOR_NAMES = {
    "XLK":"Technology","XLC":"Communication","XLY":"Consumer Disc.",
    "XLF":"Financials","XLE":"Energy","XLV":"Healthcare",
    "XLP":"Consumer Staples","XLI":"Industrials"
}

SIGNAL_COLORS = {
    "STRONG BUY":"#00e676","BUY":"#69f0ae",
    "NEUTRAL":"#ffd740","SELL":"#ff6e40","STRONG SELL":"#ff1744"
}

VIX_PAUSE = 30

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    user_input = st.text_input("Watchlist (comma-separated)", value=", ".join(DEFAULT_WATCHLIST))
    WATCHLIST  = [t.strip().upper() for t in user_input.split(",") if t.strip()]

    st.markdown("### Weights")
    w_pv  = st.slider("Price / Volume",  0, 100, 30)
    w_rs  = st.slider("Sector RS",       0, 100, 25)
    w_brd = st.slider("Breadth",         0, 100, 20)
    w_vlt = st.slider("Volatility",      0, 100, 15)
    w_yld = st.slider("Yield Curve",     0, 100, 10)

    total_w = w_pv + w_rs + w_brd + w_vlt + w_yld
    if total_w != 100:
        st.warning(f"Weights sum to {total_w}. Auto-normalising.")

    WEIGHTS = {k: v/total_w for k, v in {
        "price_vol":w_pv,"sector_rs":w_rs,"breadth":w_brd,
        "volatility":w_vlt,"yield_curve":w_yld
    }.items()}

    st.button("🔄 Run Model", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption(f"Last run: {datetime.now().strftime('%H:%M:%S')}")

# ── SHARED HELPERS ────────────────────────────────────────────────────────────

def normalise(val, lo, hi):
    return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))

def get_col(raw, ticker, field):
    try:
        if (ticker, field) in raw.columns:
            return raw[(ticker, field)].dropna()
        if field in raw.columns:
            return raw[field].dropna()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

def compute_atr(close, window=14):
    hi = close.rolling(2).max()
    lo = close.rolling(2).min()
    return (hi - lo).rolling(window).mean()

def compute_dms_at(i, close, volume, vix, spy, sect, t10, t5, weights):
    try:
        c  = close.iloc[:i+1]; v = volume.iloc[:i+1]
        vx = vix.iloc[:i+1];   sp = spy.iloc[:i+1]; sc = sect.iloc[:i+1]
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
        rs = normalise((1+sr)/((1+er) if (1+er)!=0 else 1), 0.70, 1.30)

        return round(
            weights["price_vol"]*pv + weights["sector_rs"]*rs +
            weights["breadth"]*brd  + weights["volatility"]*vlt +
            weights["yield_curve"]*yld, 2)
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_all(tickers):
    end = datetime.today(); start = end - timedelta(days=90)
    return yf.download(list(tickers), start=start, end=end,
                       auto_adjust=True, progress=False, group_by="ticker")

def get_signal(score):
    if score >= 75: return "STRONG BUY",  "Full position"
    if score >= 60: return "BUY",          "Half position"
    if score >= 40: return "NEUTRAL",      "Hold / watch"
    if score >= 25: return "SELL",         "Reduce position"
    return "STRONG SELL", "Exit / short"

def atr_stop(close_s):
    hi  = close_s.rolling(2).max()
    lo  = close_s.rolling(2).min()
    atr = (hi - lo).rolling(14).mean().iloc[-1]
    return round(close_s.iloc[-1] - 1.5*atr, 2), round(atr, 2)

# ── DAILY MODEL ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def run_model(watchlist_key, weights_key):
    all_tix = tuple(set(list(watchlist_key) + list(SECTOR_MAP.values()) +
                        ["SPY","^VIX","^TNX","^FVX"]))
    raw     = fetch_all(all_tix)
    vix_s   = get_col(raw, "^VIX", "Close")
    spy_s   = get_col(raw, "SPY",  "Close")
    vix_now = float(vix_s.iloc[-1]) if not vix_s.empty else 20.0
    regime  = vix_now < VIX_PAUSE
    rows    = []

    for ticker in watchlist_key:
        close_s = get_col(raw, ticker, "Close")
        vol_s   = get_col(raw, ticker, "Volume")
        sect_s  = get_col(raw, SECTOR_MAP.get(ticker,"SPY"), "Close")
        if len(close_s) < 25: continue
        try:
            i   = len(close_s) - 1
            t10 = get_col(raw, "^TNX", "Close")
            t5  = get_col(raw, "^FVX", "Close")
            dms = compute_dms_at(i, close_s, vol_s, vix_s, spy_s, sect_s, t10, t5, WEIGHTS)
            if dms is None: continue

            # sub-scores for breakdown tab
            pv = vlt = brd = yld = rs = 0.0
            c  = close_s; v = vol_s
            avg_vol   = v.rolling(20).mean().iloc[-1]
            vol_ratio = v.iloc[-1]/avg_vol if avg_vol>0 else 1.0
            pv  = normalise((c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*vol_ratio,-0.05,0.05)
            vix_score = 100-normalise(vix_s.iloc[-1],10,40)
            hi_c=c.rolling(2).max();lo_c=c.rolling(2).min()
            atr_pct=((hi_c-lo_c)/c).rolling(14).mean().iloc[-1]*100
            vlt = (vix_score+(100-normalise(atr_pct,0.5,5.0)))/2
            brd = normalise(spy_s.iloc[-1]/spy_s.rolling(50).mean().iloc[-1],0.90,1.10)
            try:
                yld = normalise(t10.iloc[-1]-t5.iloc[-1],-0.5,2.0)
            except Exception:
                yld = 50.0
            w2=min(20,len(c)-1)
            sr=(c.iloc[-1]/c.iloc[-w2])-1
            er=(sect_s.iloc[-1]/sect_s.iloc[-w2])-1
            rs=normalise((1+sr)/((1+er) if (1+er)!=0 else 1),0.70,1.30)

            sig, act  = get_signal(dms)
            stop, atr = atr_stop(close_s)
            rows.append({
                "Ticker":ticker,"Price":round(close_s.iloc[-1],2),
                "DMS":round(dms,1),"Signal":sig,"Action":act,
                "Stop":stop,"ATR":atr,
                "Sector":SECTOR_NAMES.get(SECTOR_MAP.get(ticker,""),"—"),
                "PV":round(pv,1),"Volatility":round(vlt,1),
                "Breadth":round(brd,1),"Yield":round(yld,1),"SectorRS":round(rs,1),
                "Close":close_s,"blocked":not regime and "BUY" in sig
            })
        except Exception:
            continue

    if not rows:
        st.error("No data returned. Check watchlist or try again shortly.")
        st.stop()
    return pd.DataFrame(rows).sort_values("DMS",ascending=False), vix_now, regime

# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def run_backtest(ticker, start_str, end_str, capital, weights_key):
    weights = dict(zip(
        ["price_vol","sector_rs","breadth","volatility","yield_curve"],
        weights_key
    ))
    all_tix = list(set([ticker, SECTOR_MAP.get(ticker,"SPY"),
                        "SPY","^VIX","^TNX","^FVX"]))
    raw = yf.download(all_tix, start=start_str, end=end_str,
                      auto_adjust=True, progress=False, group_by="ticker")

    close  = get_col(raw, ticker,                    "Close")
    volume = get_col(raw, ticker,                    "Volume")
    vix    = get_col(raw, "^VIX",                    "Close")
    spy    = get_col(raw, "SPY",                     "Close")
    sect   = get_col(raw, SECTOR_MAP.get(ticker,"SPY"), "Close")
    t10    = get_col(raw, "^TNX",                    "Close")
    t5     = get_col(raw, "^FVX",                    "Close")

    if len(close) < 60:
        return None, None, None, None

    idx  = close.index
    vix  = vix.reindex(idx, method="ffill")
    spy  = spy.reindex(idx, method="ffill")
    sect = sect.reindex(idx,method="ffill")
    t10  = t10.reindex(idx, method="ffill")
    t5   = t5.reindex(idx,  method="ffill")
    atr_s = compute_atr(close)

    cap = float(capital); pos = 0; entry_px = 0.0; stop_px = 0.0
    trades = []; equity = []; dms_list = []

    for i in range(len(close)):
        price = close.iloc[i]; date = idx[i]
        atr   = atr_s.iloc[i] if not pd.isna(atr_s.iloc[i]) else 0
        dms   = compute_dms_at(i, close, volume, vix, spy, sect, t10, t5, weights)
        dms_list.append(dms)

        if pos > 0 and price <= stop_px:
            pnl = (price - entry_px) * pos; cap += pos * price
            trades.append({"Date":date,"Action":"STOP","Price":round(price,2),
                            "Shares":pos,"PnL":round(pnl,2)}); pos = 0

        if dms is not None:
            if dms >= 60 and pos == 0 and vix.iloc[i] < 30:
                shares = int(cap * 0.10 / price)
                if shares > 0:
                    cap -= shares * price; pos = shares
                    entry_px = price; stop_px = price - 1.5 * atr
                    trades.append({"Date":date,"Action":"BUY","Price":round(price,2),
                                   "Shares":shares,"PnL":0})
            elif dms <= 40 and pos > 0:
                pnl = (price - entry_px) * pos; cap += pos * price
                trades.append({"Date":date,"Action":"SELL","Price":round(price,2),
                                "Shares":pos,"PnL":round(pnl,2)}); pos = 0

        equity.append(cap + pos * price)

    if pos > 0:
        price = close.iloc[-1]; pnl = (price-entry_px)*pos; cap += pos*price
        trades.append({"Date":idx[-1],"Action":"CLOSE","Price":round(price,2),
                        "Shares":pos,"PnL":round(pnl,2)})

    equity_s  = pd.Series(equity, index=idx)
    dms_s     = pd.Series(dms_list, index=idx)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
                    columns=["Date","Action","Price","Shares","PnL"])
    return trades_df, equity_s, dms_s, close

def calc_metrics(trades_df, equity_s, close, capital):
    total_ret = (equity_s.iloc[-1] - capital) / capital * 100
    bh_ret    = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
    daily_ret = equity_s.pct_change().dropna()
    sharpe    = (daily_ret.mean()/daily_ret.std()*np.sqrt(252)
                 if daily_ret.std()>0 else 0)
    max_dd    = ((equity_s - equity_s.cummax()) / equity_s.cummax()*100).min()
    closed    = trades_df[trades_df["Action"].isin(["SELL","STOP","CLOSE"])]
    wins      = closed[closed["PnL"]>0];  losses = closed[closed["PnL"]<=0]
    n         = len(closed)
    hit_rate  = len(wins)/n*100 if n>0 else 0
    avg_win   = wins["PnL"].mean()   if len(wins)>0   else 0
    avg_loss  = losses["PnL"].mean() if len(losses)>0 else 0
    pf        = abs(avg_win/avg_loss) if avg_loss!=0 else 0
    return {
        "Total Return %": round(total_ret,1),
        "Buy & Hold %":   round(bh_ret,1),
        "Alpha %":        round(total_ret-bh_ret,1),
        "Sharpe Ratio":   round(sharpe,2),
        "Max Drawdown %": round(max_dd,1),
        "Total Trades":   n,
        "Win Rate %":     round(hit_rate,1),
        "Profit Factor":  round(pf,2),
        "Final Value $":  round(equity_s.iloc[-1],0),
    }

# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-bar">
    <h2 style="color:white;margin:0">📊 Equity Quant Model</h2>
    <p style="color:#90caf9;margin:4px 0 0">Daily scoring dashboard — 5-factor signal engine</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD DAILY DATA ───────────────────────────────────────────────────────────

with st.spinner("Fetching market data..."):
    df, vix_now, regime = run_model(tuple(WATCHLIST), str(WEIGHTS))

# ── KPIs ──────────────────────────────────────────────────────────────────────

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("VIX", f"{vix_now:.1f}", delta="Risk-ON" if regime else "Risk-OFF",
          delta_color="normal" if regime else "inverse")
k2.metric("Stocks Scored", len(df))
k3.metric("Buy Signals",  len(df[df["Signal"].isin(["STRONG BUY","BUY"])]))
k4.metric("Sell Signals", len(df[df["Signal"].isin(["STRONG SELL","SELL"])]))
k5.metric("Avg DMS",      f"{df['DMS'].mean():.1f}")

if not regime:
    st.warning("🚨 VIX > 30 — Regime filter active. All BUY signals are blocked.")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Signals", "📊 Score Breakdown", "🗺️ Sector Map",
    "📈 Price Charts", "🔁 Backtester"
])

# ── TAB 1: SIGNALS ────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Ranked Signal Table")
    display_df = df[["Ticker","Price","DMS","Signal","Action","Stop","ATR","Sector"]].copy()

    def colour_signal(val):
        return f"color: {SIGNAL_COLORS.get(val,'white')}; font-weight: bold"

    def colour_dms(val):
        bg = "#1b5e20" if val>=75 else "#2e7d32" if val>=60 else \
             "#f57f17" if val>=40 else "#bf360c" if val>=25 else "#b71c1c"
        return f"background-color:{bg}; color:white; font-weight:bold"

    st.dataframe(
        display_df.style.map(colour_signal,subset=["Signal"]).map(colour_dms,subset=["DMS"]),
        use_container_width=True, height=420
    )
    st.download_button("⬇️ Download CSV", display_df.to_csv(index=False),
                       f"signals_{datetime.today().strftime('%Y%m%d')}.csv","text/csv")

# ── TAB 2: SCORE BREAKDOWN ────────────────────────────────────────────────────

with tab2:
    st.subheader("Sub-Score Breakdown per Stock")
    sub = df[["Ticker","PV","Volatility","Breadth","Yield","SectorRS"]].melt(
        id_vars="Ticker", var_name="Factor", value_name="Score")
    fig_bar = px.bar(sub, x="Ticker", y="Score", color="Factor", barmode="group",
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     title="Sub-scores by Stock", height=400)
    fig_bar.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",font_color="white")
    st.plotly_chart(fig_bar, use_container_width=True)

    sel = st.selectbox("Radar — select stock", df["Ticker"].tolist())
    row = df[df["Ticker"]==sel].iloc[0]
    cats = ["Price/Vol","Volatility","Breadth","Yield","Sector RS"]
    vals = [row["PV"],row["Volatility"],row["Breadth"],row["Yield"],row["SectorRS"]]
    fig_radar = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself", fillcolor="rgba(0,150,255,0.2)",
        line=dict(color="#2196F3",width=2)
    ))
    fig_radar.update_layout(
        polar=dict(bgcolor="#1c2030",
                   radialaxis=dict(visible=True,range=[0,100],color="white"),
                   angularaxis=dict(color="white")),
        paper_bgcolor="#0e1117",font_color="white",
        title=f"{sel} — Factor Radar (DMS: {row['DMS']})",height=400)
    st.plotly_chart(fig_radar, use_container_width=True)

# ── TAB 3: SECTOR MAP ─────────────────────────────────────────────────────────

with tab3:
    st.subheader("Sector Heatmap")
    s_avg = df.groupby("Sector")["DMS"].mean().reset_index()
    s_cnt = df.groupby("Sector")["Ticker"].count().reset_index()
    s_df  = s_avg.merge(s_cnt,on="Sector"); s_df.columns=["Sector","Avg DMS","Count"]
    fig_heat = px.treemap(s_df, path=["Sector"], values="Count",
                          color="Avg DMS", color_continuous_scale="RdYlGn",
                          range_color=[0,100], height=420)
    fig_heat.update_layout(paper_bgcolor="#0e1117",font_color="white")
    st.plotly_chart(fig_heat, use_container_width=True)

    fig_sc = px.scatter(df, x="Sector", y="DMS", color="Signal",
                        color_discrete_map=SIGNAL_COLORS, text="Ticker",
                        size=[20]*len(df), height=380)
    fig_sc.update_traces(textposition="top center")
    fig_sc.add_hline(y=60,line_dash="dash",line_color="#69f0ae",annotation_text="Buy zone")
    fig_sc.add_hline(y=40,line_dash="dash",line_color="#ff6e40",annotation_text="Sell zone")
    fig_sc.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",font_color="white")
    st.plotly_chart(fig_sc, use_container_width=True)

# ── TAB 4: PRICE CHARTS ───────────────────────────────────────────────────────

with tab4:
    st.subheader("Price Chart with Stop-Loss")
    sel2  = st.selectbox("Select stock", df["Ticker"].tolist(), key="chart2")
    row2  = df[df["Ticker"]==sel2].iloc[0]
    cls   = row2["Close"]
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=cls.index,y=cls.values,name="Price",
                               line=dict(color="#2196F3",width=2)))
    fig_p.add_trace(go.Scatter(x=cls.index,y=cls.rolling(20).mean(),name="20MA",
                               line=dict(color="#ffd740",width=1.5,dash="dash")))
    fig_p.add_trace(go.Scatter(x=cls.index,y=cls.rolling(50).mean(),name="50MA",
                               line=dict(color="#ff6e40",width=1.5,dash="dot")))
    fig_p.add_hline(y=row2["Stop"],line_dash="dash",line_color="#ff1744",
                    annotation_text=f"Stop: {row2['Stop']}",
                    annotation_font_color="#ff1744")
    fig_p.update_layout(
        title=f"{sel2} — {row2['Signal']}  |  DMS: {row2['DMS']}  |  ${row2['Price']}",
        paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",font_color="white",height=450)
    st.plotly_chart(fig_p, use_container_width=True)

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=row2["DMS"], delta={"reference":50},
        gauge={"axis":{"range":[0,100]},"bar":{"color":SIGNAL_COLORS.get(row2["Signal"],"white")},
               "steps":[{"range":[0,25],"color":"#b71c1c"},{"range":[25,40],"color":"#bf360c"},
                         {"range":[40,60],"color":"#f57f17"},{"range":[60,75],"color":"#2e7d32"},
                         {"range":[75,100],"color":"#1b5e20"}],
               "threshold":{"line":{"color":"white","width":3},"value":row2["DMS"]}},
        title={"text":f"{sel2} Daily Market Score","font":{"color":"white"}}
    ))
    fig_g.update_layout(paper_bgcolor="#0e1117",font_color="white",height=300)
    st.plotly_chart(fig_g, use_container_width=True)

# ── TAB 5: BACKTESTER ─────────────────────────────────────────────────────────

with tab5:
    st.subheader("🔁 Backtest a Stock")
    st.caption("Test how the model's signals would have performed historically on any stock.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bt_ticker  = st.selectbox("Stock", WATCHLIST, key="bt_tick")
    with c2:
        bt_start   = st.date_input("Start date", value=datetime(2022,1,1))
    with c3:
        bt_end     = st.date_input("End date",   value=datetime(2024,12,31))
    with c4:
        bt_capital = st.number_input("Starting capital ($)", value=10000, step=1000)

    run_bt = st.button("▶️  Run Backtest", type="primary")

    if run_bt:
        with st.spinner(f"Backtesting {bt_ticker} from {bt_start} to {bt_end}..."):
            w_tuple = tuple(WEIGHTS.values())
            trades_df, equity_s, dms_s, close_s = run_backtest(
                bt_ticker, str(bt_start), str(bt_end), bt_capital, w_tuple
            )

        if equity_s is None:
            st.error("Not enough data for this ticker / date range. Try a wider range.")
        else:
            metrics = calc_metrics(trades_df, equity_s, close_s, bt_capital)

            # ── Metric cards ──
            st.markdown("#### Performance Summary")
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Total Return",  f"{metrics['Total Return %']}%",
                      delta=f"{metrics['Alpha %']:+.1f}% vs B&H")
            m2.metric("Buy & Hold",    f"{metrics['Buy & Hold %']}%")
            m3.metric("Sharpe Ratio",  metrics["Sharpe Ratio"],
                      delta="Good" if metrics["Sharpe Ratio"]>=1 else "Weak",
                      delta_color="normal" if metrics["Sharpe Ratio"]>=1 else "inverse")
            m4.metric("Max Drawdown",  f"{metrics['Max Drawdown %']}%")
            m5.metric("Win Rate",      f"{metrics['Win Rate %']}%")
            m6.metric("Final Value",   f"${metrics['Final Value $']:,.0f}")

            st.markdown("---")

            # ── Chart 1: Price + signals ──
            st.markdown("#### Price Chart with Buy / Sell Signals")
            fig_bt1 = go.Figure()
            fig_bt1.add_trace(go.Scatter(x=close_s.index, y=close_s.values,
                                         name="Price", line=dict(color="#2196F3",width=1.5)))
            fig_bt1.add_trace(go.Scatter(x=close_s.index, y=close_s.rolling(20).mean(),
                                         name="20MA", line=dict(color="#ffd740",width=1,dash="dash")))
            if not trades_df.empty:
                buys  = trades_df[trades_df["Action"]=="BUY"]
                sells = trades_df[trades_df["Action"].isin(["SELL","CLOSE"])]
                stops = trades_df[trades_df["Action"]=="STOP"]
                if len(buys):
                    fig_bt1.add_trace(go.Scatter(x=buys["Date"],  y=buys["Price"],
                        mode="markers", marker=dict(symbol="triangle-up",size=12,color="#00e676"),
                        name="Buy"))
                if len(sells):
                    fig_bt1.add_trace(go.Scatter(x=sells["Date"], y=sells["Price"],
                        mode="markers", marker=dict(symbol="triangle-down",size=12,color="#ffd740"),
                        name="Sell"))
                if len(stops):
                    fig_bt1.add_trace(go.Scatter(x=stops["Date"], y=stops["Price"],
                        mode="markers", marker=dict(symbol="x",size=12,color="#ff1744"),
                        name="Stop"))
            fig_bt1.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                                  font_color="white",height=420,
                                  title=f"{bt_ticker} — Signals ({bt_start} → {bt_end})")
            st.plotly_chart(fig_bt1, use_container_width=True)

            # ── Chart 2: Equity curve ──
            st.markdown("#### Equity Curve vs Buy & Hold")
            bh = (close_s / close_s.iloc[0]) * bt_capital
            fig_bt2 = go.Figure()
            fig_bt2.add_trace(go.Scatter(x=equity_s.index, y=equity_s.values,
                                         name="Model", line=dict(color="#00e676",width=2)))
            fig_bt2.add_trace(go.Scatter(x=bh.index, y=bh.values,
                                         name="Buy & Hold", line=dict(color="#90caf9",width=2,dash="dash")))
            fig_bt2.add_hline(y=bt_capital, line_dash="dot", line_color="#555")
            fig_bt2.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                                  font_color="white",height=380,
                                  yaxis_title="Portfolio Value ($)",xaxis_title="Date")
            st.plotly_chart(fig_bt2, use_container_width=True)

            # ── Chart 3: DMS over time ──
            st.markdown("#### Daily Market Score Over Time")
            dms_clean = dms_s.dropna()
            fig_bt3   = go.Figure()
            fig_bt3.add_trace(go.Scatter(x=dms_clean.index, y=dms_clean.values,
                                         name="DMS", line=dict(color="#ffd740",width=1)))
            fig_bt3.add_hrect(y0=60, y1=100, fillcolor="#00e676", opacity=0.07, line_width=0)
            fig_bt3.add_hrect(y0=0,  y1=40,  fillcolor="#ff1744", opacity=0.07, line_width=0)
            fig_bt3.add_hline(y=60, line_dash="dash", line_color="#00e676",
                              annotation_text="Buy (60)")
            fig_bt3.add_hline(y=40, line_dash="dash", line_color="#ff6e40",
                              annotation_text="Sell (40)")
            fig_bt3.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                                  font_color="white",height=300,
                                  yaxis=dict(range=[0,100]),xaxis_title="Date",yaxis_title="DMS")
            st.plotly_chart(fig_bt3, use_container_width=True)

            # ── Trade log ──
            st.markdown("#### Trade Log")
            if not trades_df.empty:
                def colour_action(val):
                    c = {"BUY":"#00e676","SELL":"#ffd740","STOP":"#ff1744","CLOSE":"#90caf9"}.get(val,"white")
                    return f"color:{c}; font-weight:bold"
                def colour_pnl(val):
                    return f"color:{'#00e676' if val>0 else '#ff1744' if val<0 else 'white'}"
                st.dataframe(
                    trades_df.style.map(colour_action,subset=["Action"]).map(colour_pnl,subset=["PnL"]),
                    use_container_width=True, height=300
                )
                st.download_button("⬇️ Download Trade Log",
                                   trades_df.to_csv(index=False),
                                   f"trades_{bt_ticker}_{bt_start}_{bt_end}.csv","text/csv")
            else:
                st.info("No trades were generated in this period.")

# ── FOOTER ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("⚠️ For informational purposes only. Not financial advice. Always apply your own judgement before trading.")
