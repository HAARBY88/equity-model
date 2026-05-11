"""
Equity Quant Dashboard — Simplified 3-Filter System
Gate 1: Market healthy? (VIX + SPY 200MA combined)
Gate 2: Stock signal strong? (DMS threshold)
Gate 3: Portfolio not overexposed? (max positions + sector cap)
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

st.set_page_config(page_title="Equity Quant Model", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .header-bar {
        background: linear-gradient(90deg, #1a237e, #283593);
        padding: 18px 24px; border-radius: 10px; margin-bottom: 20px;
    }
    .gate-pass { background:#1b5e20; border-radius:8px; padding:12px 16px; text-align:center; }
    .gate-fail { background:#b71c1c; border-radius:8px; padding:12px 16px; text-align:center; }
    .gate-label { color:white; font-size:13px; margin-bottom:4px; }
    .gate-value { color:white; font-size:20px; font-weight:800; }
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

WEIGHTS = {
    "price_vol":0.30,"sector_rs":0.25,"breadth":0.20,
    "volatility":0.15,"yield_curve":0.10
}

# ── SIDEBAR — only 3 settings ─────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    user_input = st.text_input("Watchlist (comma-separated)",
                               value=", ".join(DEFAULT_WATCHLIST))
    WATCHLIST = [t.strip().upper() for t in user_input.split(",") if t.strip()]

    st.markdown("---")
    st.markdown("### 3 Filters")

    st.markdown("**Gate 1 — Market**")
    vix_level = st.select_slider(
        "Market sensitivity",
        options=["Relaxed (VIX<35)", "Normal (VIX<25)", "Strict (VIX<20)"],
        value="Normal (VIX<25)"
    )
    VIX_LIMIT = {"Relaxed (VIX<35)":35, "Normal (VIX<25)":25, "Strict (VIX<20)":20}[vix_level]

    st.markdown("**Gate 2 — Signal**")
    signal_level = st.select_slider(
        "Signal strength required",
        options=["Moderate (DMS≥60)", "Strong (DMS≥70)", "Very Strong (DMS≥75)"],
        value="Strong (DMS≥70)"
    )
    DMS_LIMIT = {"Moderate (DMS≥60)":60, "Strong (DMS≥70)":70, "Very Strong (DMS≥75)":75}[signal_level]

    st.markdown("**Gate 3 — Exposure**")
    MAX_POSITIONS = st.slider("Max open positions", 3, 15, 8)

    st.markdown("---")
    st.button("🔄 Refresh", use_container_width=True, type="primary")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

# ── HELPERS ───────────────────────────────────────────────────────────────────

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
    return (close.rolling(2).max() - close.rolling(2).min()).rolling(window).mean()

def compute_dms(close_s, vol_s, vix_s, spy_s, sect_s, t10, t5):
    avg_vol   = vol_s.rolling(20).mean().iloc[-1]
    vol_ratio = vol_s.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
    pct_chg   = (close_s.iloc[-1] - close_s.iloc[-2]) / close_s.iloc[-2]
    pv        = normalise(pct_chg * vol_ratio, -0.05, 0.05)
    vix_score = 100 - normalise(vix_s.iloc[-1], 10, 40)
    hi=close_s.rolling(2).max(); lo=close_s.rolling(2).min()
    atr_pct = ((hi-lo)/close_s).rolling(14).mean().iloc[-1]*100
    vlt = (vix_score + (100-normalise(atr_pct, 0.5, 5.0))) / 2
    brd = normalise(spy_s.iloc[-1]/spy_s.rolling(50).mean().iloc[-1], 0.90, 1.10)
    try:    yld = normalise(t10.iloc[-1]-t5.iloc[-1], -0.5, 2.0)
    except: yld = 50.0
    w = min(20, len(close_s)-1)
    sr = (close_s.iloc[-1]/close_s.iloc[-w])-1
    er = (sect_s.iloc[-1]/sect_s.iloc[-w])-1
    rs = normalise((1+sr)/((1+er) if (1+er)!=0 else 1), 0.70, 1.30)
    return round(
        WEIGHTS["price_vol"]*pv + WEIGHTS["sector_rs"]*rs +
        WEIGHTS["breadth"]*brd  + WEIGHTS["volatility"]*vlt +
        WEIGHTS["yield_curve"]*yld, 2)

def get_signal(score):
    if score >= 75: return "STRONG BUY"
    if score >= 60: return "BUY"
    if score >= 40: return "NEUTRAL"
    if score >= 25: return "SELL"
    return "STRONG SELL"

@st.cache_data(ttl=300)
def fetch_all(tickers):
    end = datetime.today(); start = end - timedelta(days=120)
    return yf.download(list(tickers), start=start, end=end,
                       auto_adjust=True, progress=False, group_by="ticker")

# ── GATE 1: MARKET HEALTH ─────────────────────────────────────────────────────

def check_market(vix_s, spy_s):
    vix_now  = float(vix_s.iloc[-1])
    spy_now  = float(spy_s.iloc[-1])
    spy_200  = float(spy_s.rolling(200).mean().iloc[-1])
    vix_ok   = vix_now < VIX_LIMIT
    spy_ok   = spy_now > spy_200
    market_ok= vix_ok and spy_ok
    return market_ok, vix_now, spy_now, spy_200, vix_ok, spy_ok

# ── MAIN MODEL ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def run_model(wl_key, dms_lim, vix_lim, max_pos):
    all_tix = tuple(set(list(wl_key)+list(SECTOR_MAP.values())+["SPY","^VIX","^TNX","^FVX"]))
    raw     = fetch_all(all_tix)
    vix_s   = get_col(raw,"^VIX","Close"); spy_s=get_col(raw,"SPY","Close")
    t10     = get_col(raw,"^TNX","Close"); t5   =get_col(raw,"^FVX","Close")

    market_ok, vix_now, spy_now, spy_200, vix_ok, spy_ok = check_market(vix_s, spy_s)

    rows=[]; sector_counts={}; position_count=0

    for ticker in wl_key:
        close_s = get_col(raw,ticker,"Close")
        vol_s   = get_col(raw,ticker,"Volume")
        sect_s  = get_col(raw,SECTOR_MAP.get(ticker,"SPY"),"Close")
        if len(close_s) < 25: continue
        try:
            dms     = compute_dms(close_s,vol_s,vix_s,spy_s,sect_s,t10,t5)
            signal  = get_signal(dms)
            sector  = SECTOR_MAP.get(ticker,"—")
            sec_cnt = sector_counts.get(sector,0)

            # Gate 2: signal strong enough
            g2 = dms >= dms_lim
            # Gate 3: not overexposed (max 2 per sector, max total positions)
            g3 = sec_cnt < 2 and position_count < max_pos

            eligible = market_ok and g2 and g3
            if eligible:
                sector_counts[sector] = sec_cnt + 1
                position_count += 1

            atr   = compute_atr(close_s).iloc[-1]
            stop  = round(close_s.iloc[-1] - 1.5*atr, 2)
            trail = round(close_s.iloc[-1] - 2.0*atr, 2)
            size  = 0.15 if signal=="STRONG BUY" else 0.08

            rows.append({
                "Ticker":   ticker,
                "Price":    round(close_s.iloc[-1], 2),
                "DMS":      round(dms, 1),
                "Signal":   signal,
                "Trade?":   "✅ BUY" if eligible else "—",
                "Stop":     stop,
                "Trail":    trail,
                "Size":     f"{size*100:.0f}%",
                "Sector":   SECTOR_NAMES.get(sector,"—"),
                "G1 Mkt":   "✅" if market_ok else "❌",
                "G2 DMS":   "✅" if g2 else "❌",
                "G3 Exp":   "✅" if g3 else "❌",
                "Close":    close_s,
            })
        except: continue

    if not rows:
        st.error("No data returned. Check watchlist."); st.stop()

    return (pd.DataFrame(rows).sort_values("DMS", ascending=False),
            vix_now, spy_now, spy_200, vix_ok, spy_ok, market_ok)

# ── BACKTEST ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def run_backtest(ticker, start_str, end_str, capital, dms_lim, vix_lim):
    all_tix = list(set([ticker, SECTOR_MAP.get(ticker,"SPY"),
                        "SPY","^VIX","^TNX","^FVX"]))
    raw = yf.download(all_tix, start=start_str, end=end_str,
                      auto_adjust=True, progress=False, group_by="ticker")

    close  = get_col(raw,ticker,"Close");   volume=get_col(raw,ticker,"Volume")
    vix    = get_col(raw,"^VIX","Close");   spy   =get_col(raw,"SPY","Close")
    sect   = get_col(raw,SECTOR_MAP.get(ticker,"SPY"),"Close")
    t10    = get_col(raw,"^TNX","Close");   t5    =get_col(raw,"^FVX","Close")
    if len(close)<60: return None,None,None,None

    idx = close.index
    vix =vix.reindex(idx,method="ffill");  spy =spy.reindex(idx,method="ffill")
    sect=sect.reindex(idx,method="ffill"); t10=t10.reindex(idx,method="ffill")
    t5  =t5.reindex(idx,method="ffill")
    atr_s = compute_atr(close)

    cap=float(capital); pos=0; entry_px=0.0; stop_px=0.0; highest=0.0
    trades=[]; equity=[]; dms_list=[]

    for i in range(len(close)):
        price=close.iloc[i]; date=idx[i]
        atr  =atr_s.iloc[i] if not pd.isna(atr_s.iloc[i]) else 0
        c=close.iloc[:i+1]; v=volume.iloc[:i+1]
        vx=vix.iloc[:i+1]; sp=spy.iloc[:i+1]; sc=sect.iloc[:i+1]
        t_10=t10.iloc[:i+1]; t_5=t5.iloc[:i+1]

        if len(c)<25:
            equity.append(cap+pos*price); dms_list.append(None); continue

        try:    dms = compute_dms(c,v,vx,sp,sc,t_10,t_5)
        except: dms = None
        dms_list.append(dms)

        # Gate 1
        vix_ok = vx.iloc[-1] < vix_lim
        spy_ok = sp.iloc[-1] > sp.rolling(200).mean().iloc[-1]
        mkt_ok = vix_ok and spy_ok

        # Trailing stop
        if pos > 0:
            highest = max(price, highest)
            stop_px = max(stop_px, highest - 2.0*atr)

        if pos > 0 and price <= stop_px:
            pnl=(price-entry_px)*pos; cap+=pos*price
            trades.append({"Date":date,"Action":"STOP","Price":round(price,2),
                            "Shares":pos,"PnL":round(pnl,2)}); pos=0

        if dms is not None:
            size_pct = 0.15 if dms>=75 else 0.08
            if dms>=dms_lim and pos==0 and mkt_ok:
                shares=int(cap*size_pct/price)
                if shares>0:
                    cap-=shares*price; pos=shares; entry_px=price
                    highest=price; stop_px=price-1.5*atr
                    trades.append({"Date":date,"Action":"BUY","Price":round(price,2),
                                   "Shares":shares,"PnL":0})
            elif dms<=40 and pos>0:
                pnl=(price-entry_px)*pos; cap+=pos*price
                trades.append({"Date":date,"Action":"SELL","Price":round(price,2),
                                "Shares":pos,"PnL":round(pnl,2)}); pos=0

        equity.append(cap+pos*price)

    if pos>0:
        price=close.iloc[-1]; pnl=(price-entry_px)*pos; cap+=pos*price
        trades.append({"Date":idx[-1],"Action":"CLOSE","Price":round(price,2),
                        "Shares":pos,"PnL":round(pnl,2)})

    equity_s =pd.Series(equity,index=idx)
    dms_s    =pd.Series(dms_list,index=idx)
    trades_df=pd.DataFrame(trades) if trades else pd.DataFrame(
               columns=["Date","Action","Price","Shares","PnL"])
    return trades_df, equity_s, dms_s, close

def calc_metrics(trades_df, equity_s, close, capital):
    tr=(equity_s.iloc[-1]-capital)/capital*100
    bh=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100
    dr=equity_s.pct_change().dropna()
    sh=(dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0)
    dd=((equity_s-equity_s.cummax())/equity_s.cummax()*100).min()
    cl=trades_df[trades_df["Action"].isin(["SELL","STOP","CLOSE"])]
    wi=cl[cl["PnL"]>0]; lo=cl[cl["PnL"]<=0]; n=len(cl)
    hr=len(wi)/n*100 if n>0 else 0
    aw=wi["PnL"].mean() if len(wi)>0 else 0
    al=lo["PnL"].mean() if len(lo)>0 else 0
    pf=abs(aw/al) if al!=0 else 0
    return {"Total Return %":round(tr,1),"Buy & Hold %":round(bh,1),
            "Alpha %":round(tr-bh,1),"Sharpe Ratio":round(sh,2),
            "Max Drawdown %":round(dd,1),"Total Trades":n,
            "Win Rate %":round(hr,1),"Profit Factor":round(pf,2),
            "Final Value $":round(equity_s.iloc[-1],0)}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

with st.spinner("Loading..."):
    df, vix_now, spy_now, spy_200, vix_ok, spy_ok, market_ok = run_model(
        tuple(WATCHLIST), DMS_LIMIT, VIX_LIMIT, MAX_POSITIONS)

# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-bar">
    <h2 style="color:white;margin:0">📊 Equity Quant Model</h2>
    <p style="color:#90caf9;margin:4px 0 0">3-filter signal system — simple, consistent, repeatable</p>
</div>
""", unsafe_allow_html=True)

# ── 3 GATE STATUS CARDS ───────────────────────────────────────────────────────

st.markdown("### Today's Gates")
g1, g2, g3, g4 = st.columns(4)

with g1:
    cls = "gate-pass" if market_ok else "gate-fail"
    icon = "✅" if market_ok else "❌"
    spy_diff = ((spy_now/spy_200)-1)*100
    st.markdown(f"""
    <div class="{cls}">
        <div class="gate-label">Gate 1 — Market Health</div>
        <div class="gate-value">{icon} {"OPEN" if market_ok else "CLOSED"}</div>
        <div class="gate-label">VIX {vix_now:.1f} {'✅' if vix_ok else '❌'} &nbsp;|&nbsp;
        SPY {spy_diff:+.1f}% vs 200MA {'✅' if spy_ok else '❌'}</div>
    </div>""", unsafe_allow_html=True)

with g2:
    eligible = df[df["Trade?"]=="✅ BUY"]
    has_sig  = len(eligible) > 0
    cls = "gate-pass" if has_sig else "gate-fail"
    st.markdown(f"""
    <div class="{cls}">
        <div class="gate-label">Gate 2 — Signal Strength</div>
        <div class="gate-value">{len(eligible)} stock{"s" if len(eligible)!=1 else ""} ≥ {DMS_LIMIT}</div>
        <div class="gate-label">Threshold: DMS ≥ {DMS_LIMIT}</div>
    </div>""", unsafe_allow_html=True)

with g3:
    open_pos = len(eligible)
    at_cap   = open_pos >= MAX_POSITIONS
    cls = "gate-fail" if at_cap else "gate-pass"
    st.markdown(f"""
    <div class="{cls}">
        <div class="gate-label">Gate 3 — Exposure</div>
        <div class="gate-value">{open_pos} / {MAX_POSITIONS} slots</div>
        <div class="gate-label">Max 2 per sector &nbsp;|&nbsp;
        {"AT CAPACITY" if at_cap else "Slots available"}</div>
    </div>""", unsafe_allow_html=True)

with g4:
    all_pass = market_ok and has_sig and not at_cap
    cls = "gate-pass" if all_pass else "gate-fail"
    action = "TRADE TODAY" if all_pass else "STAND ASIDE"
    st.markdown(f"""
    <div class="{cls}">
        <div class="gate-label">Overall Decision</div>
        <div class="gate-value">{"✅" if all_pass else "❌"} {action}</div>
        <div class="gate-label">All 3 gates must be green</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────

tab1,tab2,tab3,tab4 = st.tabs(["📋 Signals","📈 Charts","🗺️ Sectors","🔁 Backtest"])

# ── TAB 1: SIGNALS ────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Stock Rankings")

    # Highlight eligible trades at top
    if len(eligible) > 0:
        st.success(f"✅ **Trade candidates today:** {', '.join(eligible['Ticker'].tolist())}")
    else:
        st.info("No stocks pass all 3 gates today. Stand aside.")

    disp = df[["Ticker","Price","DMS","Signal","Trade?","Stop","Trail","Size",
               "Sector","G1 Mkt","G2 DMS","G3 Exp"]].copy()

    def csig(v): return f"color:{SIGNAL_COLORS.get(v,'white')};font-weight:bold"
    def cdms(v):
        bg="#1b5e20" if v>=75 else "#2e7d32" if v>=60 else \
           "#f57f17" if v>=40 else "#bf360c" if v>=25 else "#b71c1c"
        return f"background-color:{bg};color:white;font-weight:bold"
    def ctrade(v):
        return "color:#00e676;font-weight:bold" if "BUY" in str(v) else "color:#555"

    st.dataframe(
        disp.style
            .map(csig,   subset=["Signal"])
            .map(cdms,   subset=["DMS"])
            .map(ctrade, subset=["Trade?"]),
        use_container_width=True, height=420)

    st.download_button("⬇️ Download CSV", disp.to_csv(index=False),
                       f"signals_{datetime.today().strftime('%Y%m%d')}.csv","text/csv")

# ── TAB 2: CHARTS ─────────────────────────────────────────────────────────────

with tab2:
    sel = st.selectbox("Select stock", df["Ticker"].tolist())
    row = df[df["Ticker"]==sel].iloc[0]
    cls = row["Close"]

    # Price chart
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=cls.index,y=cls.values,name="Price",
                             line=dict(color="#2196F3",width=2)))
    fig.add_trace(go.Scatter(x=cls.index,y=cls.rolling(20).mean(),name="20MA",
                             line=dict(color="#ffd740",width=1.5,dash="dash")))
    fig.add_trace(go.Scatter(x=cls.index,y=cls.rolling(50).mean(),name="50MA",
                             line=dict(color="#ff6e40",width=1.5,dash="dot")))
    fig.add_hline(y=row["Stop"], line_dash="dash",line_color="#ff1744",
                  annotation_text=f"Stop {row['Stop']}",annotation_font_color="#ff1744")
    fig.add_hline(y=row["Trail"],line_dash="dot", line_color="#ff6e40",
                  annotation_text=f"Trail {row['Trail']}",annotation_font_color="#ff6e40")
    fig.update_layout(
        title=f"{sel}  |  DMS {row['DMS']}  |  {row['Signal']}  |  Trade? {row['Trade?']}",
        paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",font_color="white",height=420)
    st.plotly_chart(fig,use_container_width=True)

    # DMS gauge
    fig_g=go.Figure(go.Indicator(
        mode="gauge+number",value=row["DMS"],
        gauge={"axis":{"range":[0,100]},
               "bar":{"color":SIGNAL_COLORS.get(row["Signal"],"white")},
               "steps":[{"range":[0,25],"color":"#b71c1c"},
                         {"range":[25,40],"color":"#bf360c"},
                         {"range":[40,60],"color":"#f57f17"},
                         {"range":[60,75],"color":"#2e7d32"},
                         {"range":[75,100],"color":"#1b5e20"}],
               "threshold":{"line":{"color":"white","width":3},"value":DMS_LIMIT}},
        title={"text":f"{sel} — DMS (line = buy threshold {DMS_LIMIT})",
               "font":{"color":"white"}}))
    fig_g.update_layout(paper_bgcolor="#0e1117",font_color="white",height=280)
    st.plotly_chart(fig_g,use_container_width=True)

# ── TAB 3: SECTORS ────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Sector Overview")
    sa=df.groupby("Sector")["DMS"].mean().reset_index()
    sc_=df.groupby("Sector")["Ticker"].count().reset_index()
    sdf=sa.merge(sc_,on="Sector"); sdf.columns=["Sector","Avg DMS","Count"]
    fig_h=px.treemap(sdf,path=["Sector"],values="Count",color="Avg DMS",
                     color_continuous_scale="RdYlGn",range_color=[0,100],height=400)
    fig_h.update_layout(paper_bgcolor="#0e1117",font_color="white")
    st.plotly_chart(fig_h,use_container_width=True)

    fig_sc=px.scatter(df,x="Sector",y="DMS",color="Signal",
                      color_discrete_map=SIGNAL_COLORS,text="Ticker",
                      size=[20]*len(df),height=360)
    fig_sc.update_traces(textposition="top center")
    fig_sc.add_hline(y=DMS_LIMIT,line_dash="dash",line_color="#69f0ae",
                     annotation_text=f"Buy threshold ({DMS_LIMIT})")
    fig_sc.add_hline(y=40,line_dash="dash",line_color="#ff6e40",
                     annotation_text="Sell (40)")
    fig_sc.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",font_color="white")
    st.plotly_chart(fig_sc,use_container_width=True)

# ── TAB 4: BACKTEST ───────────────────────────────────────────────────────────

with tab4:
    st.subheader("🔁 Backtest a Stock")
    st.caption("Tests using your current 3-filter settings.")

    c1,c2,c3,c4=st.columns(4)
    with c1: bt_ticker  = st.selectbox("Stock",WATCHLIST,key="bt")
    with c2: bt_start   = st.date_input("Start",value=datetime(2022,1,1))
    with c3: bt_end     = st.date_input("End",  value=datetime(2024,12,31))
    with c4: bt_capital = st.number_input("Capital ($)",value=10000,step=1000)

    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner(f"Backtesting {bt_ticker}..."):
            trades_df,equity_s,dms_s,close_s = run_backtest(
                bt_ticker,str(bt_start),str(bt_end),bt_capital,DMS_LIMIT,VIX_LIMIT)

        if equity_s is None:
            st.error("Not enough data. Try a wider date range.")
        else:
            m=calc_metrics(trades_df,equity_s,close_s,bt_capital)

            m1,m2,m3,m4,m5,m6=st.columns(6)
            m1.metric("Total Return", f"{m['Total Return %']}%",
                      delta=f"{m['Alpha %']:+.1f}% vs B&H")
            m2.metric("Buy & Hold",   f"{m['Buy & Hold %']}%")
            m3.metric("Sharpe",       m["Sharpe Ratio"],
                      delta="Good" if m["Sharpe Ratio"]>=1 else "Weak",
                      delta_color="normal" if m["Sharpe Ratio"]>=1 else "inverse")
            m4.metric("Max Drawdown", f"{m['Max Drawdown %']}%")
            m5.metric("Win Rate",     f"{m['Win Rate %']}%")
            m6.metric("Final Value",  f"${m['Final Value $']:,.0f}")

            st.markdown("---")

            # Price + signals
            fig1=go.Figure()
            fig1.add_trace(go.Scatter(x=close_s.index,y=close_s.values,name="Price",
                                      line=dict(color="#2196F3",width=1.5)))
            fig1.add_trace(go.Scatter(x=close_s.index,y=close_s.rolling(20).mean(),
                                      name="20MA",line=dict(color="#ffd740",width=1,dash="dash")))
            if not trades_df.empty:
                for action,sym,col in [("BUY","triangle-up","#00e676"),
                                        ("SELL","triangle-down","#ffd740"),
                                        ("STOP","x","#ff1744")]:
                    t=trades_df[trades_df["Action"]==action]
                    if len(t):
                        fig1.add_trace(go.Scatter(x=t["Date"],y=t["Price"],mode="markers",
                            marker=dict(symbol=sym,size=12,color=col),name=action))
            fig1.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                               font_color="white",height=400,
                               title=f"{bt_ticker} — Price & Signals")
            st.plotly_chart(fig1,use_container_width=True)

            # Equity curve
            bh=(close_s/close_s.iloc[0])*bt_capital
            fig2=go.Figure()
            fig2.add_trace(go.Scatter(x=equity_s.index,y=equity_s.values,
                                      name="Model",line=dict(color="#00e676",width=2)))
            fig2.add_trace(go.Scatter(x=bh.index,y=bh.values,name="Buy & Hold",
                                      line=dict(color="#90caf9",width=2,dash="dash")))
            fig2.add_hline(y=bt_capital,line_dash="dot",line_color="#555")
            fig2.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                               font_color="white",height=340,
                               yaxis_title="Value ($)",xaxis_title="Date")
            st.plotly_chart(fig2,use_container_width=True)

            # DMS over time
            dms_c=dms_s.dropna()
            fig3=go.Figure()
            fig3.add_trace(go.Scatter(x=dms_c.index,y=dms_c.values,name="DMS",
                                      line=dict(color="#ffd740",width=1)))
            fig3.add_hrect(y0=DMS_LIMIT,y1=100,fillcolor="#00e676",opacity=0.07,line_width=0)
            fig3.add_hrect(y0=0,y1=40,fillcolor="#ff1744",opacity=0.07,line_width=0)
            fig3.add_hline(y=DMS_LIMIT,line_dash="dash",line_color="#00e676",
                           annotation_text=f"Buy ({DMS_LIMIT})")
            fig3.add_hline(y=40,line_dash="dash",line_color="#ff6e40",
                           annotation_text="Sell (40)")
            fig3.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                               font_color="white",height=260,
                               yaxis=dict(range=[0,100]),xaxis_title="Date")
            st.plotly_chart(fig3,use_container_width=True)

            # Trade log
            if not trades_df.empty:
                st.markdown("#### Trade Log")
                def ca(v):
                    c={"BUY":"#00e676","SELL":"#ffd740","STOP":"#ff1744","CLOSE":"#90caf9"}.get(v,"white")
                    return f"color:{c};font-weight:bold"
                def cp(v): return f"color:{'#00e676' if v>0 else '#ff1744' if v<0 else 'white'}"
                st.dataframe(trades_df.style.map(ca,subset=["Action"]).map(cp,subset=["PnL"]),
                             use_container_width=True,height=280)
                st.download_button("⬇️ Download Trades",trades_df.to_csv(index=False),
                                   f"trades_{bt_ticker}.csv","text/csv")
            else:
                st.info("No trades generated in this period.")

st.markdown("---")
st.caption("⚠️ For informational purposes only. Not financial advice.")
