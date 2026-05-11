"""
Equity Quant Dashboard — Enhanced for Better Sharpe Ratio
Improvements:
  1. Higher entry threshold (DMS >= 70)
  2. SPY 200MA regime filter
  3. Sector exposure cap (max 2 per sector)
  4. Conviction-based position sizing
  5. Trailing stop
  6. Time-based stop (10 days)
  7. Dual timeframe filter (weekly + daily)
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

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    user_input = st.text_input("Watchlist (comma-separated)", value=", ".join(DEFAULT_WATCHLIST))
    WATCHLIST  = [t.strip().upper() for t in user_input.split(",") if t.strip()]

    st.markdown("### Signal Weights")
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
        "volatility":w_vlt,"yield_curve":w_yld}.items()}

    st.markdown("### Sharpe Filters")
    buy_threshold    = st.slider("Buy threshold (DMS)",  50, 85, 70)
    sell_threshold   = st.slider("Sell threshold (DMS)", 20, 55, 40)
    vix_pause        = st.slider("VIX pause level",      15, 40, 30)
    max_sector_slots = st.slider("Max stocks per sector", 1, 5, 2)
    time_stop_days   = st.slider("Time stop (days)",      5, 30, 10)
    trail_mult       = st.slider("Trailing stop (× ATR)", 1.0, 4.0, 2.0, 0.5)
    use_spy_filter   = st.checkbox("SPY 200MA regime filter", value=True)
    use_dual_tf      = st.checkbox("Dual timeframe filter (weekly)", value=True)

    st.button("🔄 Run Model", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption(f"Last run: {datetime.now().strftime('%H:%M:%S')}")

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

def compute_dms(close_s, vol_s, vix_s, spy_s, sect_s, t10, t5, weights):
    avg_vol   = vol_s.rolling(20).mean().iloc[-1]
    vol_ratio = vol_s.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
    pct_chg   = (close_s.iloc[-1] - close_s.iloc[-2]) / close_s.iloc[-2]
    pv        = normalise(pct_chg * vol_ratio, -0.05, 0.05)
    vix_score = 100 - normalise(vix_s.iloc[-1], 10, 40)
    hi = close_s.rolling(2).max(); lo = close_s.rolling(2).min()
    atr_pct = ((hi-lo)/close_s).rolling(14).mean().iloc[-1]*100
    vlt = (vix_score + (100-normalise(atr_pct, 0.5, 5.0)))/2
    brd = normalise(spy_s.iloc[-1]/spy_s.rolling(50).mean().iloc[-1], 0.90, 1.10)
    try:    yld = normalise(t10.iloc[-1]-t5.iloc[-1], -0.5, 2.0)
    except: yld = 50.0
    w2 = min(20, len(close_s)-1)
    sr = (close_s.iloc[-1]/close_s.iloc[-w2])-1
    er = (sect_s.iloc[-1]/sect_s.iloc[-w2])-1
    rs = normalise((1+sr)/((1+er) if (1+er)!=0 else 1), 0.70, 1.30)
    return round(
        weights["price_vol"]*pv + weights["sector_rs"]*rs +
        weights["breadth"]*brd  + weights["volatility"]*vlt +
        weights["yield_curve"]*yld, 2), pv, vlt, brd, yld, rs

def compute_weekly_dms(close_s, vol_s, vix_s, spy_s, sect_s, t10, t5, weights):
    try:
        wc = close_s.resample("W").last(); wv = vol_s.resample("W").sum()
        wvix=vix_s.resample("W").last();   wspy=spy_s.resample("W").last()
        wsct=sect_s.resample("W").last();  wt10=t10.resample("W").last()
        wt5=t5.resample("W").last()
        if len(wc)<10: return None
        val,*_ = compute_dms(wc,wv,wvix,wspy,wsct,wt10,wt5,weights)
        return val
    except: return None

def get_signal(score, buy_thr):
    if score >= 75:       return "STRONG BUY"
    if score >= buy_thr:  return "BUY"
    if score >= 40:       return "NEUTRAL"
    if score >= 25:       return "SELL"
    return "STRONG SELL"

def regime_check(vix_s, spy_s, vix_pause, use_spy):
    vix_now = vix_s.iloc[-1]
    if vix_now >= vix_pause: return False, f"VIX {vix_now:.1f} ≥ {vix_pause}"
    if use_spy:
        spy_200 = spy_s.rolling(200).mean().iloc[-1]
        if spy_s.iloc[-1] < spy_200:
            return False, f"SPY below 200MA"
    return True, "Risk-ON"

@st.cache_data(ttl=300)
def fetch_all(tickers):
    end = datetime.today(); start = end - timedelta(days=120)
    return yf.download(list(tickers), start=start, end=end,
                       auto_adjust=True, progress=False, group_by="ticker")

# ── DAILY MODEL ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def run_model(wl_key, w_key, buy_thr, sell_thr, vix_p, max_sec, use_spy, use_dual):
    all_tix = tuple(set(list(wl_key)+list(SECTOR_MAP.values())+["SPY","^VIX","^TNX","^FVX"]))
    raw     = fetch_all(all_tix)
    vix_s   = get_col(raw,"^VIX","Close"); spy_s=get_col(raw,"SPY","Close")
    t10     = get_col(raw,"^TNX","Close"); t5  =get_col(raw,"^FVX","Close")
    vix_now = float(vix_s.iloc[-1]) if not vix_s.empty else 20.0
    ok, reason = regime_check(vix_s, spy_s, vix_p, use_spy)
    rows=[]; sector_counts={}

    for ticker in wl_key:
        close_s = get_col(raw,ticker,"Close"); vol_s=get_col(raw,ticker,"Volume")
        sect_s  = get_col(raw,SECTOR_MAP.get(ticker,"SPY"),"Close")
        if len(close_s)<25: continue
        try:
            dms,pv,vlt,brd,yld,rs = compute_dms(close_s,vol_s,vix_s,spy_s,sect_s,t10,t5,WEIGHTS)
            w_dms = compute_weekly_dms(close_s,vol_s,vix_s,spy_s,sect_s,t10,t5,WEIGHTS) if use_dual else 999
            dual_ok  = (w_dms is None or w_dms >= 55)
            sector   = SECTOR_MAP.get(ticker,"—")
            sec_cnt  = sector_counts.get(sector,0)
            sec_ok   = sec_cnt < max_sec
            signal   = get_signal(dms, buy_thr)
            atr      = compute_atr(close_s).iloc[-1]
            stop     = round(close_s.iloc[-1]-1.5*atr,2)
            trail    = round(close_s.iloc[-1]-trail_mult*atr,2)
            size_pct = 0.15 if signal=="STRONG BUY" else 0.08
            eligible = ok and dms>=buy_thr and dual_ok and sec_ok
            if eligible: sector_counts[sector]=sec_cnt+1
            rows.append({
                "Ticker":ticker,"Price":round(close_s.iloc[-1],2),
                "DMS":round(dms,1),"Weekly DMS":round(w_dms,1) if w_dms and w_dms!=999 else "—",
                "Signal":signal,"Eligible":"✅" if eligible else "❌",
                "Stop":stop,"Trail":trail,"ATR":round(atr,2),
                "Size%":f"{size_pct*100:.0f}%",
                "Sector":SECTOR_NAMES.get(sector,"—"),
                "PV":round(pv,1),"Volatility":round(vlt,1),
                "Breadth":round(brd,1),"Yield":round(yld,1),"SectorRS":round(rs,1),
                "Close":close_s,
                "Regime":"✅" if ok else "❌",
                "DualTF":"✅" if dual_ok else "❌",
                "SecSlot":"✅" if sec_ok else "❌",
            })
        except: continue

    if not rows:
        st.error("No data returned. Check watchlist or try again."); st.stop()
    return pd.DataFrame(rows).sort_values("DMS",ascending=False), vix_now, ok, reason

# ── BACKTEST ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def run_backtest(ticker, start_str, end_str, capital, w_key,
                 buy_thr, sell_thr, vix_p, use_spy, use_dual, t_mult, time_stop):
    weights = dict(zip(["price_vol","sector_rs","breadth","volatility","yield_curve"], w_key))
    all_tix = list(set([ticker,SECTOR_MAP.get(ticker,"SPY"),"SPY","^VIX","^TNX","^FVX"]))
    raw = yf.download(all_tix,start=start_str,end=end_str,
                      auto_adjust=True,progress=False,group_by="ticker")

    close  = get_col(raw,ticker,"Close");  volume=get_col(raw,ticker,"Volume")
    vix    = get_col(raw,"^VIX","Close");  spy   =get_col(raw,"SPY","Close")
    sect   = get_col(raw,SECTOR_MAP.get(ticker,"SPY"),"Close")
    t10    = get_col(raw,"^TNX","Close");  t5    =get_col(raw,"^FVX","Close")
    if len(close)<60: return None,None,None,None

    idx  = close.index
    for s in [vix,spy,sect,t10,t5]: s = s.reindex(idx,method="ffill")
    vix=vix.reindex(idx,method="ffill"); spy=spy.reindex(idx,method="ffill")
    sect=sect.reindex(idx,method="ffill"); t10=t10.reindex(idx,method="ffill")
    t5=t5.reindex(idx,method="ffill")
    atr_s = compute_atr(close)

    cap=float(capital); pos=0; entry_px=0.0; stop_px=0.0
    highest=0.0; days_held=0
    trades=[]; equity=[]; dms_list=[]

    for i in range(len(close)):
        price=close.iloc[i]; date=idx[i]
        atr  =atr_s.iloc[i] if not pd.isna(atr_s.iloc[i]) else 0
        c=close.iloc[:i+1]; v=volume.iloc[:i+1]
        vx=vix.iloc[:i+1];  sp=spy.iloc[:i+1]; sc=sect.iloc[:i+1]
        t_10=t10.iloc[:i+1];t_5=t5.iloc[:i+1]
        if len(c)<25:
            equity.append(cap+pos*price); dms_list.append(None); continue

        try:
            dms,*_ = compute_dms(c,v,vx,sp,sc,t_10,t_5,weights)
        except:
            dms=None
        dms_list.append(dms)

        # Regime
        ok,_ = regime_check(vx,sp,vix_p,use_spy)

        # Weekly DMS
        if use_dual and dms is not None:
            try:
                w_dms,*_ = compute_dms(
                    c.resample("W").last(), v.resample("W").sum(),
                    vx.resample("W").last(), sp.resample("W").last(),
                    sc.resample("W").last(), t_10.resample("W").last(),
                    t_5.resample("W").last(), weights)
                dual_ok = w_dms >= 55
            except: dual_ok=True
        else: dual_ok=True

        # Trailing stop update
        if pos > 0:
            highest,stop_px = max(price,highest), max(stop_px, highest - t_mult*atr)
            days_held += 1

        # Stop-loss
        if pos > 0 and price <= stop_px:
            pnl=( price-entry_px)*pos; cap+=pos*price
            trades.append({"Date":date,"Action":"STOP","Price":round(price,2),
                            "Shares":pos,"PnL":round(pnl,2)}); pos=0; days_held=0

        # Time stop
        if pos > 0 and days_held >= time_stop and (price-entry_px)*pos < 0:
            pnl=(price-entry_px)*pos; cap+=pos*price
            trades.append({"Date":date,"Action":"TIME STOP","Price":round(price,2),
                            "Shares":pos,"PnL":round(pnl,2)}); pos=0; days_held=0

        if dms is not None:
            size_pct = 0.15 if dms>=75 else 0.08
            if dms>=buy_thr and pos==0 and ok and dual_ok:
                shares=int(cap*size_pct/price)
                if shares>0:
                    cap-=shares*price; pos=shares; entry_px=price
                    highest=price; stop_px=price-1.5*atr
                    trades.append({"Date":date,"Action":"BUY","Price":round(price,2),
                                   "Shares":shares,"PnL":0})
                    days_held=0
            elif dms<=sell_thr and pos>0:
                pnl=(price-entry_px)*pos; cap+=pos*price
                trades.append({"Date":date,"Action":"SELL","Price":round(price,2),
                                "Shares":pos,"PnL":round(pnl,2)}); pos=0; days_held=0

        equity.append(cap+pos*price)

    if pos>0:
        price=close.iloc[-1]; pnl=(price-entry_px)*pos; cap+=pos*price
        trades.append({"Date":idx[-1],"Action":"CLOSE","Price":round(price,2),
                        "Shares":pos,"PnL":round(pnl,2)})

    equity_s =pd.Series(equity,index=idx)
    dms_s    =pd.Series(dms_list,index=idx)
    trades_df=pd.DataFrame(trades) if trades else pd.DataFrame(
               columns=["Date","Action","Price","Shares","PnL"])
    return trades_df,equity_s,dms_s,close

def calc_metrics(trades_df,equity_s,close,capital):
    total_ret=(equity_s.iloc[-1]-capital)/capital*100
    bh_ret   =(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100
    dr=equity_s.pct_change().dropna()
    sharpe=(dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0)
    max_dd=((equity_s-equity_s.cummax())/equity_s.cummax()*100).min()
    closed=trades_df[trades_df["Action"].isin(["SELL","STOP","TIME STOP","CLOSE"])]
    wins=closed[closed["PnL"]>0]; losses=closed[closed["PnL"]<=0]
    n=len(closed); hr=len(wins)/n*100 if n>0 else 0
    aw=wins["PnL"].mean() if len(wins)>0 else 0
    al=losses["PnL"].mean() if len(losses)>0 else 0
    pf=abs(aw/al) if al!=0 else 0
    return {"Total Return %":round(total_ret,1),"Buy & Hold %":round(bh_ret,1),
            "Alpha %":round(total_ret-bh_ret,1),"Sharpe Ratio":round(sharpe,2),
            "Max Drawdown %":round(max_dd,1),"Total Trades":n,
            "Win Rate %":round(hr,1),"Profit Factor":round(pf,2),
            "Final Value $":round(equity_s.iloc[-1],0)}

# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-bar">
    <h2 style="color:white;margin:0">📊 Equity Quant Model</h2>
    <p style="color:#90caf9;margin:4px 0 0">Enhanced — 5-factor engine with Sharpe optimisation</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Fetching market data..."):
    df, vix_now, regime_on, regime_reason = run_model(
        tuple(WATCHLIST), str(WEIGHTS), buy_threshold, sell_threshold,
        vix_pause, max_sector_slots, use_spy_filter, use_dual_tf)

# ── KPIs ──────────────────────────────────────────────────────────────────────

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("VIX", f"{vix_now:.1f}", delta=regime_reason,
          delta_color="normal" if regime_on else "inverse")
k2.metric("Stocks Scored", len(df))
k3.metric("Eligible Today", len(df[df["Eligible"]=="✅"]))
k4.metric("Buy Signals", len(df[df["Signal"].isin(["STRONG BUY","BUY"])]))
k5.metric("Sell Signals", len(df[df["Signal"].isin(["STRONG SELL","SELL"])]))
k6.metric("Avg DMS", f"{df['DMS'].mean():.1f}")

if not regime_on:
    st.warning(f"🚨 Regime filter active — {regime_reason}. No new buys.")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📋 Signals","📊 Score Breakdown","🗺️ Sector Map","📈 Price Charts","🔁 Backtester"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Ranked Signal Table")
    disp = df[["Ticker","Price","DMS","Weekly DMS","Signal","Eligible",
               "Stop","Trail","Size%","Sector","Regime","DualTF","SecSlot"]].copy()

    def csig(v): return f"color:{SIGNAL_COLORS.get(v,'white')};font-weight:bold"
    def cdms(v):
        bg="#1b5e20" if v>=75 else "#2e7d32" if v>=60 else \
           "#f57f17" if v>=40 else "#bf360c" if v>=25 else "#b71c1c"
        return f"background-color:{bg};color:white;font-weight:bold"

    st.dataframe(disp.style.map(csig,subset=["Signal"]).map(cdms,subset=["DMS"]),
                 use_container_width=True, height=420)
    st.download_button("⬇️ Download CSV", disp.to_csv(index=False),
                       f"signals_{datetime.today().strftime('%Y%m%d')}.csv","text/csv")

    st.markdown("#### Filter Legend")
    fc1,fc2,fc3,fc4 = st.columns(4)
    fc1.info("**Regime** — VIX < pause AND SPY > 200MA")
    fc2.info("**DualTF** — Weekly DMS ≥ 55")
    fc3.info(f"**SecSlot** — Sector count < {max_sector_slots}")
    fc4.info(f"**DMS** — Score ≥ {buy_threshold}")

# ── TAB 2 ─────────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Sub-Score Breakdown")
    sub = df[["Ticker","PV","Volatility","Breadth","Yield","SectorRS"]].melt(
        id_vars="Ticker",var_name="Factor",value_name="Score")
    fig_bar=px.bar(sub,x="Ticker",y="Score",color="Factor",barmode="group",
                   color_discrete_sequence=px.colors.qualitative.Bold,height=400)
    fig_bar.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",font_color="white")
    st.plotly_chart(fig_bar,use_container_width=True)

    sel=st.selectbox("Radar — select stock",df["Ticker"].tolist())
    row=df[df["Ticker"]==sel].iloc[0]
    cats=["Price/Vol","Volatility","Breadth","Yield","Sector RS"]
    vals=[row["PV"],row["Volatility"],row["Breadth"],row["Yield"],row["SectorRS"]]
    fig_r=go.Figure(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],
        fill="toself",fillcolor="rgba(0,150,255,0.2)",line=dict(color="#2196F3",width=2)))
    fig_r.update_layout(polar=dict(bgcolor="#1c2030",
        radialaxis=dict(visible=True,range=[0,100],color="white"),
        angularaxis=dict(color="white")),
        paper_bgcolor="#0e1117",font_color="white",
        title=f"{sel} — Factor Radar (DMS: {row['DMS']})",height=400)
    st.plotly_chart(fig_r,use_container_width=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Sector Heatmap")
    sa=df.groupby("Sector")["DMS"].mean().reset_index()
    sc_=df.groupby("Sector")["Ticker"].count().reset_index()
    sdf=sa.merge(sc_,on="Sector"); sdf.columns=["Sector","Avg DMS","Count"]
    fig_h=px.treemap(sdf,path=["Sector"],values="Count",color="Avg DMS",
                     color_continuous_scale="RdYlGn",range_color=[0,100],height=420)
    fig_h.update_layout(paper_bgcolor="#0e1117",font_color="white")
    st.plotly_chart(fig_h,use_container_width=True)

    fig_sc=px.scatter(df,x="Sector",y="DMS",color="Signal",
                      color_discrete_map=SIGNAL_COLORS,text="Ticker",
                      size=[20]*len(df),height=380)
    fig_sc.update_traces(textposition="top center")
    fig_sc.add_hline(y=buy_threshold,line_dash="dash",line_color="#69f0ae",
                     annotation_text=f"Buy ({buy_threshold})")
    fig_sc.add_hline(y=sell_threshold,line_dash="dash",line_color="#ff6e40",
                     annotation_text=f"Sell ({sell_threshold})")
    fig_sc.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",font_color="white")
    st.plotly_chart(fig_sc,use_container_width=True)

# ── TAB 4 ─────────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("Price Chart with Stops")
    sel2=st.selectbox("Select stock",df["Ticker"].tolist(),key="chart2")
    row2=df[df["Ticker"]==sel2].iloc[0]; cls=row2["Close"]
    fig_p=go.Figure()
    fig_p.add_trace(go.Scatter(x=cls.index,y=cls.values,name="Price",
                               line=dict(color="#2196F3",width=2)))
    fig_p.add_trace(go.Scatter(x=cls.index,y=cls.rolling(20).mean(),name="20MA",
                               line=dict(color="#ffd740",width=1.5,dash="dash")))
    fig_p.add_trace(go.Scatter(x=cls.index,y=cls.rolling(50).mean(),name="50MA",
                               line=dict(color="#ff6e40",width=1.5,dash="dot")))
    fig_p.add_hline(y=row2["Stop"],line_dash="dash",line_color="#ff1744",
                    annotation_text=f"Stop: {row2['Stop']}",annotation_font_color="#ff1744")
    fig_p.add_hline(y=row2["Trail"],line_dash="dot",line_color="#ff6e40",
                    annotation_text=f"Trail: {row2['Trail']}",annotation_font_color="#ff6e40")
    fig_p.update_layout(title=f"{sel2} — {row2['Signal']}  |  DMS: {row2['DMS']}  |  ${row2['Price']}",
        paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",font_color="white",height=450)
    st.plotly_chart(fig_p,use_container_width=True)

    fig_g=go.Figure(go.Indicator(mode="gauge+number+delta",value=row2["DMS"],
        delta={"reference":50},
        gauge={"axis":{"range":[0,100]},"bar":{"color":SIGNAL_COLORS.get(row2["Signal"],"white")},
               "steps":[{"range":[0,25],"color":"#b71c1c"},{"range":[25,40],"color":"#bf360c"},
                         {"range":[40,60],"color":"#f57f17"},{"range":[60,75],"color":"#2e7d32"},
                         {"range":[75,100],"color":"#1b5e20"}],
               "threshold":{"line":{"color":"white","width":3},"value":buy_threshold}},
        title={"text":f"{sel2} Daily Market Score","font":{"color":"white"}}))
    fig_g.update_layout(paper_bgcolor="#0e1117",font_color="white",height=300)
    st.plotly_chart(fig_g,use_container_width=True)

# ── TAB 5: BACKTESTER ─────────────────────────────────────────────────────────

with tab5:
    st.subheader("🔁 Backtest a Stock")
    st.caption("Uses all filters currently set in the sidebar.")

    c1,c2,c3,c4 = st.columns(4)
    with c1: bt_ticker  = st.selectbox("Stock", WATCHLIST, key="bt_tick")
    with c2: bt_start   = st.date_input("Start date", value=datetime(2022,1,1))
    with c3: bt_end     = st.date_input("End date",   value=datetime(2024,12,31))
    with c4: bt_capital = st.number_input("Starting capital ($)", value=10000, step=1000)

    if st.button("▶️  Run Backtest", type="primary"):
        with st.spinner(f"Backtesting {bt_ticker}..."):
            w_tuple = tuple(WEIGHTS.values())
            trades_df,equity_s,dms_s,close_s = run_backtest(
                bt_ticker, str(bt_start), str(bt_end), bt_capital, w_tuple,
                buy_threshold, sell_threshold, vix_pause,
                use_spy_filter, use_dual_tf, trail_mult, time_stop_days)

        if equity_s is None:
            st.error("Not enough data. Try a wider date range.")
        else:
            m = calc_metrics(trades_df, equity_s, close_s, bt_capital)

            # Metric cards
            st.markdown("#### Performance Summary")
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Total Return",  f"{m['Total Return %']}%",
                      delta=f"{m['Alpha %']:+.1f}% vs B&H")
            m2.metric("Buy & Hold",    f"{m['Buy & Hold %']}%")
            m3.metric("Sharpe Ratio",  m["Sharpe Ratio"],
                      delta="Good" if m["Sharpe Ratio"]>=1 else "Weak",
                      delta_color="normal" if m["Sharpe Ratio"]>=1 else "inverse")
            m4.metric("Max Drawdown",  f"{m['Max Drawdown %']}%")
            m5.metric("Win Rate",      f"{m['Win Rate %']}%")
            m6.metric("Final Value",   f"${m['Final Value $']:,.0f}")

            st.markdown("---")

            # Price + signals chart
            st.markdown("#### Price with Signals")
            fig1=go.Figure()
            fig1.add_trace(go.Scatter(x=close_s.index,y=close_s.values,name="Price",
                                      line=dict(color="#2196F3",width=1.5)))
            fig1.add_trace(go.Scatter(x=close_s.index,y=close_s.rolling(20).mean(),name="20MA",
                                      line=dict(color="#ffd740",width=1,dash="dash")))
            if not trades_df.empty:
                for action,sym,col in [
                    ("BUY","triangle-up","#00e676"),
                    ("SELL","triangle-down","#ffd740"),
                    ("STOP","x","#ff1744"),
                    ("TIME STOP","circle-x","#ff6e40")]:
                    t=trades_df[trades_df["Action"]==action]
                    if len(t): fig1.add_trace(go.Scatter(x=t["Date"],y=t["Price"],
                        mode="markers",marker=dict(symbol=sym,size=12,color=col),name=action))
            fig1.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                               font_color="white",height=420,
                               title=f"{bt_ticker} — Signals")
            st.plotly_chart(fig1,use_container_width=True)

            # Equity curve
            st.markdown("#### Equity Curve vs Buy & Hold")
            bh=(close_s/close_s.iloc[0])*bt_capital
            fig2=go.Figure()
            fig2.add_trace(go.Scatter(x=equity_s.index,y=equity_s.values,name="Model",
                                      line=dict(color="#00e676",width=2)))
            fig2.add_trace(go.Scatter(x=bh.index,y=bh.values,name="Buy & Hold",
                                      line=dict(color="#90caf9",width=2,dash="dash")))
            fig2.add_hline(y=bt_capital,line_dash="dot",line_color="#555")
            fig2.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                               font_color="white",height=360,
                               yaxis_title="Value ($)",xaxis_title="Date")
            st.plotly_chart(fig2,use_container_width=True)

            # DMS chart
            st.markdown("#### DMS Over Time")
            dms_c=dms_s.dropna()
            fig3=go.Figure()
            fig3.add_trace(go.Scatter(x=dms_c.index,y=dms_c.values,name="DMS",
                                      line=dict(color="#ffd740",width=1)))
            fig3.add_hrect(y0=buy_threshold,y1=100,fillcolor="#00e676",opacity=0.07,line_width=0)
            fig3.add_hrect(y0=0,y1=sell_threshold,fillcolor="#ff1744",opacity=0.07,line_width=0)
            fig3.add_hline(y=buy_threshold,line_dash="dash",line_color="#00e676",
                           annotation_text=f"Buy ({buy_threshold})")
            fig3.add_hline(y=sell_threshold,line_dash="dash",line_color="#ff6e40",
                           annotation_text=f"Sell ({sell_threshold})")
            fig3.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                               font_color="white",height=280,
                               yaxis=dict(range=[0,100]),xaxis_title="Date")
            st.plotly_chart(fig3,use_container_width=True)

            # Trade log
            st.markdown("#### Trade Log")
            if not trades_df.empty:
                def ca(v):
                    c={"BUY":"#00e676","SELL":"#ffd740","STOP":"#ff1744",
                       "TIME STOP":"#ff6e40","CLOSE":"#90caf9"}.get(v,"white")
                    return f"color:{c};font-weight:bold"
                def cp(v): return f"color:{'#00e676' if v>0 else '#ff1744' if v<0 else 'white'}"
                st.dataframe(trades_df.style.map(ca,subset=["Action"]).map(cp,subset=["PnL"]),
                             use_container_width=True,height=300)
                st.download_button("⬇️ Download Trade Log",trades_df.to_csv(index=False),
                                   f"trades_{bt_ticker}_{bt_start}_{bt_end}.csv","text/csv")
            else:
                st.info("No trades generated in this period.")

st.markdown("---")
st.caption("⚠️ For informational purposes only. Not financial advice. Always apply your own judgement before trading.")
