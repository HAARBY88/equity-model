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

# ── STYLING ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1c2030; border-radius: 10px;
        padding: 16px 20px; margin: 4px 0;
    }
    .signal-strong-buy  { color: #00e676; font-weight: 800; }
    .signal-buy         { color: #69f0ae; font-weight: 700; }
    .signal-neutral     { color: #ffd740; font-weight: 600; }
    .signal-sell        { color: #ff6e40; font-weight: 700; }
    .signal-strong-sell { color: #ff1744; font-weight: 800; }
    .header-bar {
        background: linear-gradient(90deg, #1a237e, #283593);
        padding: 18px 24px; border-radius: 10px; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── CONFIG (mirrors model.py) ─────────────────────────────────────────────────

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
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/8a/NYSE_logo.svg", width=120)
    st.markdown("## ⚙️ Settings")

    user_input = st.text_input(
        "Watchlist (comma-separated tickers)",
        value=", ".join(DEFAULT_WATCHLIST)
    )
    WATCHLIST = [t.strip().upper() for t in user_input.split(",") if t.strip()]

    st.markdown("### Weights")
    w_pv  = st.slider("Price / Volume",  0, 100, 30)
    w_rs  = st.slider("Sector RS",       0, 100, 25)
    w_brd = st.slider("Breadth",         0, 100, 20)
    w_vlt = st.slider("Volatility",      0, 100, 15)
    w_yld = st.slider("Yield Curve",     0, 100, 10)

    total_w = w_pv + w_rs + w_brd + w_vlt + w_yld
    if total_w != 100:
        st.warning(f"Weights sum to {total_w}. They will be auto-normalised.")

    WEIGHTS = {k: v/total_w for k, v in {
        "price_vol": w_pv, "sector_rs": w_rs, "breadth": w_brd,
        "volatility": w_vlt, "yield_curve": w_yld
    }.items()}

    run_btn = st.button("🔄  Run Model", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption(f"Last run: {datetime.now().strftime('%H:%M:%S')}")

# ── DATA & SCORING FUNCTIONS (same logic as model.py) ─────────────────────────

def normalise(val, lo, hi):
    return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))

@st.cache_data(ttl=300)
def fetch_all(tickers):
    end   = datetime.today()
    start = end - timedelta(days=90)
    return yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

def get_series(raw, kind, ticker):
    try:
        return raw[kind][ticker].dropna()
    except Exception:
        return pd.Series(dtype=float)

def score_pv(close_s, vol_s):
    avg_vol   = vol_s.rolling(20).mean().iloc[-1]
    vol_ratio = vol_s.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
    pct_chg   = (close_s.iloc[-1] - close_s.iloc[-2]) / close_s.iloc[-2]
    return normalise(pct_chg * vol_ratio, -0.05, 0.05)

def score_vol(vix_s, close_s):
    vix_score = 100 - normalise(vix_s.iloc[-1], 10, 40)
    hi = close_s.rolling(2).max(); lo = close_s.rolling(2).min()
    atr_pct = ((hi - lo) / close_s).rolling(14).mean().iloc[-1] * 100
    return (vix_score + (100 - normalise(atr_pct, 0.5, 5.0))) / 2

def score_breadth(spy_close):
    ma50  = spy_close.rolling(50).mean().iloc[-1]
    return normalise(spy_close.iloc[-1] / ma50, 0.90, 1.10)

def score_yield(raw):
    try:
        t10 = get_series(raw, "Close", "^TNX").iloc[-1]
        t5  = get_series(raw, "Close", "^FVX").iloc[-1]
        return normalise(t10 - t5, -0.5, 2.0)
    except Exception:
        return 50.0

def score_rs(close_s, sect_s, window=20):
    sr = (close_s.iloc[-1] / close_s.iloc[-window]) - 1
    er = (sect_s.iloc[-1]  / sect_s.iloc[-window])  - 1
    rs = (1 + sr) / (1 + er) if (1 + er) != 0 else 1.0
    return normalise(rs, 0.70, 1.30)

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
    return round(close_s.iloc[-1] - 1.5 * atr, 2), round(atr, 2)

# ── RUN MODEL ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def run_model(watchlist_key, weights_key):
    all_tix = list(set(
        WATCHLIST + list(SECTOR_MAP.values()) + ["SPY","^VIX","^TNX","^FVX"]
    ))
    raw = fetch_all(tuple(all_tix))

    vix_s   = get_series(raw, "Close", "^VIX")
    spy_s   = get_series(raw, "Close", "SPY")
    vix_now = float(vix_s.iloc[-1]) if not vix_s.empty else 20.0
    regime  = vix_now < VIX_PAUSE

    rows = []
    for ticker in WATCHLIST:
        close_s = get_series(raw, "Close", ticker)
        vol_s   = get_series(raw, "Volume", ticker)
        sect_s  = get_series(raw, "Close", SECTOR_MAP.get(ticker, "SPY"))
        if len(close_s) < 25: continue
        try:
            pv  = score_pv(close_s, vol_s)
            vlt = score_vol(vix_s, close_s)
            brd = score_breadth(spy_s)
            yld = score_yield(raw)
            rs  = score_rs(close_s, sect_s)
            dms = (WEIGHTS["price_vol"] * pv + WEIGHTS["sector_rs"] * rs +
                   WEIGHTS["breadth"] * brd + WEIGHTS["volatility"] * vlt +
                   WEIGHTS["yield_curve"] * yld)
            sig, act = get_signal(dms)
            stop, atr = atr_stop(close_s)
            rows.append({
                "Ticker": ticker, "Price": round(close_s.iloc[-1], 2),
                "DMS": round(dms, 1), "Signal": sig, "Action": act,
                "Stop": stop, "ATR": atr,
                "Sector": SECTOR_NAMES.get(SECTOR_MAP.get(ticker,""), "—"),
                "PV": round(pv,1), "Volatility": round(vlt,1),
                "Breadth": round(brd,1), "Yield": round(yld,1), "SectorRS": round(rs,1),
                "Close": close_s, "blocked": not regime and "BUY" in sig
            })
        except Exception:
            continue

    df = pd.DataFrame(rows).sort_values("DMS", ascending=False)
    return df, vix_now, regime

# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-bar">
    <h2 style="color:white;margin:0">📊 Equity Quant Model</h2>
    <p style="color:#90caf9;margin:4px 0 0">Daily scoring dashboard — 5-factor signal engine</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

with st.spinner("Fetching market data..."):
    wl_key = ",".join(WATCHLIST)
    w_key  = str(WEIGHTS)
    df, vix_now, regime = run_model(wl_key, w_key)

# ── TOP KPIs ──────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("VIX", f"{vix_now:.1f}", delta="Risk-ON" if regime else "Risk-OFF",
          delta_color="normal" if regime else "inverse")
k2.metric("Stocks Scored", len(df))
k3.metric("Buy Signals",   len(df[df["Signal"].isin(["STRONG BUY","BUY"])]))
k4.metric("Sell Signals",  len(df[df["Signal"].isin(["STRONG SELL","SELL"])]))
k5.metric("Avg DMS",       f"{df['DMS'].mean():.1f}")

if not regime:
    st.warning("🚨 VIX > 30 — Regime filter active. All BUY signals are blocked.")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📋 Signals", "📊 Score Breakdown", "🗺️ Sector Map", "📈 Price Charts"])

# ── TAB 1: SIGNALS TABLE ──────────────────────────────────────────────────────

with tab1:
    st.subheader("Ranked Signal Table")

    display_df = df[["Ticker","Price","DMS","Signal","Action","Stop","ATR","Sector"]].copy()

    def colour_signal(val):
        c = SIGNAL_COLORS.get(val, "white")
        return f"color: {c}; font-weight: bold"

    def colour_dms(val):
        if val >= 75: bg = "#1b5e20"
        elif val >= 60: bg = "#2e7d32"
        elif val >= 40: bg = "#f57f17"
        elif val >= 25: bg = "#bf360c"
        else: bg = "#b71c1c"
        return f"background-color: {bg}; color: white; font-weight: bold"

    styled = (display_df.style
              .applymap(colour_signal, subset=["Signal"])
              .applymap(colour_dms,    subset=["DMS"]))

    st.dataframe(styled, use_container_width=True, height=420)

    # Download
    csv = display_df.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv,
                       f"signals_{datetime.today().strftime('%Y%m%d')}.csv",
                       "text/csv")

# ── TAB 2: SCORE BREAKDOWN ────────────────────────────────────────────────────

with tab2:
    st.subheader("Sub-Score Breakdown per Stock")

    sub = df[["Ticker","DMS","PV","Volatility","Breadth","Yield","SectorRS"]].copy()
    sub_melt = sub.melt(id_vars=["Ticker","DMS"],
                        value_vars=["PV","Volatility","Breadth","Yield","SectorRS"],
                        var_name="Factor", value_name="Score")

    fig_bar = px.bar(
        sub_melt, x="Ticker", y="Score", color="Factor", barmode="group",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Sub-scores by Stock (0–100 scale)",
        height=420
    )
    fig_bar.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="white", legend_title="Factor"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Radar for selected stock
    st.markdown("#### Radar Chart — Individual Stock")
    sel = st.selectbox("Select stock", df["Ticker"].tolist())
    row = df[df["Ticker"] == sel].iloc[0]
    cats = ["Price/Vol","Volatility","Breadth","Yield","Sector RS"]
    vals = [row["PV"], row["Volatility"], row["Breadth"], row["Yield"], row["SectorRS"]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(0,150,255,0.2)",
        line=dict(color="#2196F3", width=2)
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#1c2030",
            radialaxis=dict(visible=True, range=[0,100], color="white"),
            angularaxis=dict(color="white")
        ),
        paper_bgcolor="#0e1117", font_color="white",
        title=f"{sel} — Factor Radar (DMS: {row['DMS']})",
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ── TAB 3: SECTOR MAP ─────────────────────────────────────────────────────────

with tab3:
    st.subheader("Sector Heatmap")

    sector_avg = df.groupby("Sector")["DMS"].mean().reset_index()
    sector_cnt = df.groupby("Sector")["Ticker"].count().reset_index()
    sector_df  = sector_avg.merge(sector_cnt, on="Sector")
    sector_df.columns = ["Sector","Avg DMS","Count"]

    fig_heat = px.treemap(
        sector_df, path=["Sector"], values="Count",
        color="Avg DMS", color_continuous_scale="RdYlGn",
        range_color=[0,100],
        title="Sectors — size = # stocks, colour = avg DMS",
        height=420
    )
    fig_heat.update_layout(paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Scatter: DMS vs sector
    fig_scatter = px.scatter(
        df, x="Sector", y="DMS", size=[20]*len(df),
        color="Signal", color_discrete_map=SIGNAL_COLORS,
        text="Ticker", title="DMS by Sector",
        height=380
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.add_hline(y=60, line_dash="dash", line_color="#69f0ae",
                          annotation_text="Buy zone")
    fig_scatter.add_hline(y=40, line_dash="dash", line_color="#ff6e40",
                          annotation_text="Sell zone")
    fig_scatter.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1c2030",
        font_color="white", xaxis_title="Sector"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── TAB 4: PRICE CHARTS ───────────────────────────────────────────────────────

with tab4:
    st.subheader("Price Chart with Stop-Loss")

    sel2 = st.selectbox("Select stock", df["Ticker"].tolist(), key="chart2")
    row2 = df[df["Ticker"] == sel2].iloc[0]
    close_s = row2["Close"]

    fig_price = go.Figure()

    fig_price.add_trace(go.Scatter(
        x=close_s.index, y=close_s.values,
        name="Price", line=dict(color="#2196F3", width=2)
    ))
    fig_price.add_trace(go.Scatter(
        x=close_s.index, y=close_s.rolling(20).mean(),
        name="20-day MA", line=dict(color="#ffd740", width=1.5, dash="dash")
    ))
    fig_price.add_trace(go.Scatter(
        x=close_s.index, y=close_s.rolling(50).mean(),
        name="50-day MA", line=dict(color="#ff6e40", width=1.5, dash="dot")
    ))
    fig_price.add_hline(
        y=row2["Stop"], line_dash="dash", line_color="#ff1744",
        annotation_text=f"Stop: {row2['Stop']}", annotation_font_color="#ff1744"
    )

    sig_color = SIGNAL_COLORS.get(row2["Signal"], "white")
    fig_price.update_layout(
        title=f"{sel2} — {row2['Signal']}  |  DMS: {row2['DMS']}  |  Price: ${row2['Price']}",
        paper_bgcolor="#0e1117", plot_bgcolor="#1c2030",
        font_color="white", legend=dict(bgcolor="#1c2030"),
        xaxis_title="Date", yaxis_title="Price (USD)",
        height=450
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # DMS gauge
    st.markdown(f"#### {sel2} — DMS Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=row2["DMS"],
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0,100]},
            "bar":  {"color": sig_color},
            "steps":[
                {"range":[0,25],  "color":"#b71c1c"},
                {"range":[25,40], "color":"#bf360c"},
                {"range":[40,60], "color":"#f57f17"},
                {"range":[60,75], "color":"#2e7d32"},
                {"range":[75,100],"color":"#1b5e20"},
            ],
            "threshold":{"line":{"color":"white","width":3},"value":row2["DMS"]}
        },
        title={"text": f"{sel2} Daily Market Score", "font": {"color":"white"}}
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#0e1117", font_color="white", height=320
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("⚠️ For informational purposes only. Not financial advice. Always apply your own judgement before trading.")
