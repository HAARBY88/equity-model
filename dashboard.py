"""
Equity Quant Dashboard — Statistical Decision Engine
5-layer statistical model + Universe Screener
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

st.set_page_config(page_title="Equity Quant", page_icon="📊",
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

# ── UNIVERSES & MAPS ──────────────────────────────────────────────────────────

FTSE100 = [
    "SHEL.L","AZN.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L","BATS.L",
    "DGE.L","REL.L","NG.L","VOD.L","LLOY.L","BARC.L","NWG.L","IMB.L",
    "PRU.L","LSEG.L","ABF.L","CRH.L","EXPN.L","FERG.L","HIK.L","HLMA.L",
    "IHG.L","III.L","JD.L","KGF.L","LAND.L","MKS.L","MNG.L","MNDI.L",
    "OCDO.L","PSN.L","RKT.L","RMV.L","SGE.L","SMDS.L","SMIN.L","SSE.L",
    "SVT.L","TSCO.L","TW.L","AAL.L","ADM.L","AGK.L","ANTO.L","AUTO.L",
    "AV.L","AVV.L","BA.L","BEZ.L","BKG.L","BME.L","BNZL.L","BOO.L",
    "BRBY.L","BT-A.L","CCH.L","CCL.L","CNA.L","CPG.L","CRDA.L","DCC.L",
    "DLN.L","DPLM.L","EDV.L","ENT.L","FLTRF.L","FRES.L","GFS.L","GKN.L",
    "GLEN.L","GOOG.L","HAS.L","HLN.L","HMSO.L","HWDN.L","IAG.L","ICP.L",
    "INF.L","ITRK.L","JET2.L","KIE.L","LGEN.L","LMP.L","LRE.L","LSE.L",
    "MGGT.L","MICT.L","MKTO.L","MMB.L","MNKS.L","MRO.L","NXT.L","OML.L",
    "PHNX.L","POLYP.L","PPB.L","PZC.L","QQ.L","RDSA.L","RSA.L","RSW.L",
    "SBRY.L","SDR.L","SGRO.L","SJP.L","SKG.L","SKY.L","SLA.L","SN.L",
    "SPT.L","STAN.L","STJ.L","TATE.L","TCG.L","TRIG.L","TUI.L","UKW.L",
    "UTDI.L","UU.L","WG.L","WMH.L","WPP.L","WTB.L","XAR.L","ZAL.L",
]

FTSE_SECTOR_MAP = {
    "SHEL.L":"XLE","BP.L":"XLE","RIO.L":"XLB","GLEN.L":"XLB","ANTO.L":"XLB",
    "FRES.L":"XLB","AAL.L":"XLB","MRO.L":"XLE",
    "HSBA.L":"XLF","BARC.L":"XLF","LLOY.L":"XLF","NWG.L":"XLF","STAN.L":"XLF",
    "LSEG.L":"XLF","SJP.L":"XLF","LGEN.L":"XLF","PRU.L":"XLF","AV.L":"XLF",
    "RSA.L":"XLF","LRE.L":"XLF","PHNX.L":"XLF","MNG.L":"XLF","III.L":"XLF",
    "AZN.L":"XLV","GSK.L":"XLV","HIK.L":"XLV","HLN.L":"XLV","STJ.L":"XLV",
    "EXPN.L":"XLK","SGE.L":"XLK","AVV.L":"XLK","REL.L":"XLK","DPLM.L":"XLK",
    "HLMA.L":"XLK","SMIN.L":"XLK","ITRK.L":"XLK","CML.L":"XLK",
    "ULVR.L":"XLP","BATS.L":"XLP","IMB.L":"XLP","DGE.L":"XLP","ABF.L":"XLP",
    "TSCO.L":"XLP","SBRY.L":"XLP","MKS.L":"XLP","TATE.L":"XLP","CCH.L":"XLP",
    "IHG.L":"XLY","JD.L":"XLY","NXT.L":"XLY","BRBY.L":"XLY","RMV.L":"XLY",
    "AUTO.L":"XLY","OCDO.L":"XLY","TUI.L":"XLY","IAG.L":"XLY","CCL.L":"XLY",
    "CRH.L":"XLI","FERG.L":"XLI","BA.L":"XLI","RKT.L":"XLI","MNDI.L":"XLI",
    "SMDS.L":"XLI","PSN.L":"XLI","KIE.L":"XLI","BEZ.L":"XLI",
    "NG.L":"XLU","SSE.L":"XLU","SVT.L":"XLU","UU.L":"XLU","TRIG.L":"XLU",
    "VOD.L":"XLC","BT-A.L":"XLC","WPP.L":"XLC","MICT.L":"XLC",
    "LAND.L":"XLRE","SGRO.L":"XLRE","HMSO.L":"XLRE","LMP.L":"XLRE",
}

SP500 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","TSLA","LLY",
    "JPM","UNH","XOM","V","AVGO","PG","MA","JNJ","HD","COST",
    "MRK","ABBV","CVX","NFLX","BAC","KO","PEP","TMO","WMT","ADBE",
    "CRM","CSCO","MCD","ACN","ABT","ORCL","AMD","IBM","QCOM","TXN",
    "LIN","GE","PM","DHR","NEE","INTU","NOW","CAT","SPGI","BA",
    "RTX","UNP","GS","MS","T","LOW","ISRG","ELV","MDT","AMGN",
    "BLK","DE","SYK","AXP","GILD","REGN","SCHW","C","CB","ADI",
    "MMC","VRTX","ETN","SO","DUK","MO","ZTS","BSX","BDX","MDLZ",
    "CI","CL","ITW","WM","AON","CSX","EW","APD","PGR","CME",
    "NSC","USB","EMR","FDX","ECL","TGT","SHW","ICE","PLD","NKE",
    "WFC","PYPL","INTC","NFLX","HON","UPS","NEE","BMY","AMGN","CVS",
    "SBUX","F","GM","RIVN","LCID","SNAP","PINS","LYFT","UBER","ABNB",
    "DASH","COIN","HOOD","SOFI","AFRM","UPST","LC","SQ","BLOCK","SHOP",
    "SPOT","RBLX","U","MTCH","IAC","ZG","OPEN","OPENDOOR","REDFIN","Z",
    "CZR","MGM","LVS","WYNN","RCL","CCL","NCLH","MAR","HLT","H",
    "DAL","UAL","AAL","LUV","ALK","JBLU","HA","SKYW","SAVE","MESA",
    "XOM","CVX","COP","EOG","SLB","HAL","BKR","OXY","DVN","FANG",
    "MPC","PSX","VLO","HES","APA","NOV","FTI","RIG","NE","VAL",
    "JPM","BAC","WFC","C","GS","MS","USB","PNC","TFC","COF",
    "AXP","DFS","SYF","ALLY","CFG","FITB","HBAN","KEY","RF","MTB",
    "BK","STT","NTRS","TROW","BEN","IVZ","AMG","WDR","VRTS","APAM",
    "V","MA","PYPL","FIS","FISV","GPN","WEX","EVTC","PAYO","NUVEI",
    "UNH","CVS","CI","HUM","CNC","MOH","ELV","HCA","THC","UHS",
    "JNJ","PFE","MRK","ABBV","LLY","BMY","AMGN","GILD","BIIB","REGN",
    "VRTX","MRNA","BNTX","NVAX","SNY","AZN","GSK","NVO","RHHBY","MYL",
    "MDT","ABT","BSX","SYK","ISRG","ZBH","BAX","BDX","EW","HOLX",
    "TMO","DHR","A","WAT","IDXX","METTLER","BIO","QGEN","IQV","CRL",
    "AAPL","MSFT","NVDA","AMD","INTC","QCOM","TXN","MU","AMAT","LRCX",
    "KLAC","ASML","CDNS","SNPS","ANSS","PTC","EPAM","CTSH","INFY","WIT",
    "CRM","NOW","WDAY","VEEV","HUBS","DDOG","ZS","CRWD","S","PANW",
    "OKTA","SAIL","TENB","QLYS","VRNT","CYBR","FTNT","CHKP","NLOK","GEN",
    "GOOGL","META","NFLX","DIS","CMCSA","CHTR","PARA","WBD","FOX","FOXA",
    "TTWO","EA","ATVI","RBLX","U","ZNGA","PLTV","SKLZ","HUYA","IQ",
    "AMZN","SHOP","EBAY","ETSY","W","CHWY","OSTK","PRTS","CVNA","KMX",
    "HD","LOW","WSM","RH","BBBY","BIG","DDS","M","KSS","JWN",
    "WMT","TGT","COST","BJ","SFM","WINN","KR","ACI","SVU","CASY",
    "MCD","SBUX","YUM","QSR","DRI","TXRH","BJRI","CAKE","RUTH","EAT",
    "PG","KO","PEP","PM","MO","MDLZ","KHC","GIS","K","CPB",
    "CL","CHD","COTY","ELF","REVG","NWSA","NWS","NYT","GCI","MDP",
    "CAT","DE","EMR","ETN","HON","MMM","ITW","PH","ROK","DOV",
    "GE","RTX","LMT","NOC","GD","LHX","TDG","HEI","TXT","HII",
    "UPS","FDX","XPO","CHRW","EXPD","JBHT","ODFL","SAIA","ARCB","HTLD",
    "NSC","UNP","CSX","KSU","CP","CNI","WAB","TRN","GATX","RAIL",
    "NEE","DUK","SO","D","EXC","AEP","XEL","ES","WEC","DTE",
    "AWK","CWT","SJW","YORW","MSEX","ARTNA","GWRS","ARTESIAN","CCWC","PCYO",
    "AMT","PLD","CCI","EQIX","DLR","PSA","EXR","CUBE","LSI","NSA",
    "SPG","O","VICI","GLPI","LADR","MPW","PEAK","VTR","WELL","HR",
    "LIN","APD","ECL","SHW","PPG","RPM","AXTA","TROX","HUN","KWR",
    "NEM","GOLD","AEM","KGC","AG","PAAS","WPM","FNV","RGLD","OR",
    "FCX","SCCO","HBM","TGB","TECK","FM","IVN","CEIX","ARCH","METC",
]
SP100 = SP500  # alias kept for compatibility

SECTOR_MAP = {
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","GOOGL":"XLC","META":"XLC",
    "AMZN":"XLY","TSLA":"XLY","JPM":"XLF","UNH":"XLV","XOM":"XLE",
    "V":"XLF","AVGO":"XLK","PG":"XLP","MA":"XLF","JNJ":"XLV",
    "HD":"XLY","COST":"XLP","MRK":"XLV","ABBV":"XLV","CVX":"XLE",
    "NFLX":"XLC","BAC":"XLF","KO":"XLP","PEP":"XLP","TMO":"XLV",
    "WMT":"XLP","ADBE":"XLK","CRM":"XLK","CSCO":"XLK","MCD":"XLY",
    "ACN":"XLK","ABT":"XLV","ORCL":"XLK","AMD":"XLK","IBM":"XLK",
    "QCOM":"XLK","TXN":"XLK","LIN":"XLB","GE":"XLI","PM":"XLP",
    "DHR":"XLV","NEE":"XLU","INTU":"XLK","NOW":"XLK","CAT":"XLI",
    "SPGI":"XLF","BA":"XLI","RTX":"XLI","UNP":"XLI","GS":"XLF",
    "MS":"XLF","T":"XLC","LOW":"XLY","ISRG":"XLV","ELV":"XLV",
    "MDT":"XLV","AMGN":"XLV","BLK":"XLF","DE":"XLI","SYK":"XLV",
    "AXP":"XLF","GILD":"XLV","REGN":"XLV","SCHW":"XLF","C":"XLF",
    "CB":"XLF","ADI":"XLK","MMC":"XLF","VRTX":"XLV","ETN":"XLI",
    "SO":"XLU","DUK":"XLU","MO":"XLP","ZTS":"XLV","BSX":"XLV",
    "BDX":"XLV","MDLZ":"XLP","CI":"XLV","CL":"XLP","ITW":"XLI",
    "WM":"XLI","AON":"XLF","CSX":"XLI","EW":"XLV","APD":"XLB",
    "PGR":"XLF","CME":"XLF","NSC":"XLI","USB":"XLF","EMR":"XLI",
    "FDX":"XLI","ECL":"XLB","TGT":"XLY","SHW":"XLB","ICE":"XLF",
    "PLD":"XLRE","FIS":"XLK","BRK-B":"XLF","NKE":"XLY",
}

SECTOR_NAMES = {
    "XLK":"Technology","XLC":"Communication","XLY":"Consumer Disc.",
    "XLF":"Financials","XLE":"Energy","XLV":"Healthcare",
    "XLP":"Consumer Staples","XLI":"Industrials",
    "XLU":"Utilities","XLB":"Materials","XLRE":"Real Estate",
}

WEIGHTS = {
    "price_vol":0.30,"sector_rs":0.25,"breadth":0.20,
    "volatility":0.15,"yield_curve":0.10
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    user_input = st.text_input("Watchlist (for Decisions tab)",
                               value="AAPL,MSFT,NVDA,GOOGL,META,AMZN,JPM,XOM,UNH,V")
    WATCHLIST = [t.strip().upper() for t in user_input.split(",") if t.strip()]

    st.markdown("---")
    st.markdown("### Statistical thresholds")
    MIN_WIN_PROB = st.slider("Min win probability",     0.45, 0.75, 0.55, 0.01)
    MIN_Z        = st.slider("Min Z-score entry",       0.0,  2.0,  0.5,  0.1)
    MAX_Z        = st.slider("Max Z-score (overbought)",1.5,  4.0,  2.5,  0.1)
    KELLY_FRAC   = st.slider("Kelly fraction",          0.25, 1.0,  0.5,  0.05)
    MC_N         = st.select_slider("Monte Carlo runs",
                                    options=[200,500,1000,2000], value=1000)
    MAX_POS      = st.slider("Max position size %",     5, 25, 15)
    P25_FLOOR    = st.slider("Max acceptable P25 loss ($)", -2000, 0, -500, 100)

    st.markdown("---")
    st.button("🔄 Refresh", use_container_width=True, type="primary")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

# ── HELPERS ───────────────────────────────────────────────────────────────────

def normalise(val, lo, hi):
    return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))

def get_col(raw, ticker, field):
    try:
        if (ticker, field) in raw.columns: return raw[(ticker, field)].dropna()
        if field in raw.columns:           return raw[field].dropna()
        return pd.Series(dtype=float)
    except: return pd.Series(dtype=float)

@st.cache_data(ttl=300)
def fetch_data(tickers_key, days=420):
    end = datetime.today(); start = end - timedelta(days=days)
    return yf.download(list(tickers_key), start=start, end=end,
                       auto_adjust=True, progress=False, group_by="ticker")

# ── STATISTICAL FUNCTIONS ─────────────────────────────────────────────────────

def compute_dms(c, v, vix_s, spy_s, sc, t10, t5):
    avg_vol   = v.rolling(20).mean().iloc[-1]
    vol_ratio = v.iloc[-1]/avg_vol if avg_vol > 0 else 1.0
    pv        = normalise((c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*vol_ratio, -0.05, 0.05)
    vix_score = 100 - normalise(vix_s.iloc[-1], 10, 40)
    hi = c.rolling(2).max(); lo = c.rolling(2).min()
    atr_pct   = ((hi-lo)/c).rolling(14).mean().iloc[-1]*100
    vlt       = (vix_score + (100-normalise(atr_pct, 0.5, 5.0))) / 2
    brd       = normalise(spy_s.iloc[-1]/spy_s.rolling(50).mean().iloc[-1], 0.90, 1.10)
    try:    yld = normalise(t10.iloc[-1]-t5.iloc[-1], -0.5, 2.0)
    except: yld = 50.0
    w  = min(20, len(c)-1)
    sr = (c.iloc[-1]/c.iloc[-w])-1; er = (sc.iloc[-1]/sc.iloc[-w])-1
    rs = normalise((1+sr)/((1+er) if (1+er) != 0 else 1), 0.70, 1.30)
    return round(WEIGHTS["price_vol"]*pv + WEIGHTS["sector_rs"]*rs +
                 WEIGHTS["breadth"]*brd  + WEIGHTS["volatility"]*vlt +
                 WEIGHTS["yield_curve"]*yld, 2)

def bayesian_win_prob(c, dms_now, window=20, horizon=10):
    ret  = c.pct_change().dropna()
    rm   = ret.rolling(window).mean(); rs = ret.rolling(window).std()
    proxy = ((rm/rs.replace(0,np.nan)).clip(-3,3)*16.67+50).dropna()
    fwd   = c.pct_change(horizon).shift(-horizon)
    aln   = pd.concat([proxy, fwd], axis=1).dropna()
    aln.columns = ["d","f"]
    lo = max(0, dms_now-10); hi = min(100, dms_now+10)
    b  = aln[(aln["d"]>=lo)&(aln["d"]<=hi)]
    if len(b) < 10: return 0.50, len(b)
    return round(float((b["f"]>0).mean()), 3), len(b)

def classify_regime(vix_s, spy_s):
    vix    = vix_s.iloc[-1]; spy = spy_s.iloc[-1]
    spy200 = spy_s.rolling(200).mean().iloc[-1]
    above  = spy > spy200
    if above and vix < 15:    return "Bull quiet",    True,  2.1, vix, spy, spy200
    if above and vix < 25:    return "Bull volatile", True,  0.8, vix, spy, spy200
    if not above and vix < 20:return "Bear quiet",    False,-0.3, vix, spy, spy200
    return "Bear volatile", False, -2.4, vix, spy, spy200

def zscore_entry(c, window=20):
    ret = c.pct_change().dropna()
    mu  = ret.rolling(window).mean().iloc[-1]
    sd  = ret.rolling(window).std().iloc[-1]
    if sd == 0 or pd.isna(sd): return 0.0
    return round(float((ret.iloc[-1]-mu)/sd), 2)

def kelly_size(win_prob, avg_win=0.03, avg_loss=0.015):
    if avg_loss == 0: return 0.0
    b = avg_win/avg_loss; q = 1-win_prob
    return round(min(max((b*win_prob-q)/b*KELLY_FRAC, 0), MAX_POS/100), 4)

def monte_carlo(c, kelly, capital=100000, horizon=10, n=None):
    if n is None: n = MC_N
    ret    = c.pct_change().dropna().values
    entry  = c.iloc[-1]; shares = (capital*kelly)/entry
    oc     = np.array([(entry*np.prod(1+np.random.choice(ret,size=horizon,replace=True))-entry)*shares
                        for _ in range(n)])
    return (round(float(np.percentile(oc,25)),0),
            round(float(np.percentile(oc,50)),0),
            round(float(np.percentile(oc,75)),0),
            round(float((oc<0).mean()),3), oc)

def evaluate(ticker, c, v, vix_s, spy_s, sc, t10, t5):
    dms              = compute_dms(c, v, vix_s, spy_s, sc, t10, t5)
    win_prob, n_samp = bayesian_win_prob(c, dms)
    regime, trd, exp_ret, vix_now, spy_now, spy_200 = classify_regime(vix_s, spy_s)
    z                = zscore_entry(c)
    kelly            = kelly_size(win_prob)
    p25,p50,p75,ploss,mc_oc = monte_carlo(c, kelly) if kelly > 0 else (0,0,0,1.0,np.array([0]))
    l1=win_prob>=MIN_WIN_PROB; l2=trd; l3=MIN_Z<=z<=MAX_Z
    l4=kelly>0.01;             l5=p25>P25_FLOOR
    all_pass = l1 and l2 and l3 and l4 and l5
    atr = (c.rolling(2).max()-c.rolling(2).min()).rolling(14).mean().iloc[-1]
    return {
        "Ticker":ticker, "Price":round(float(c.iloc[-1]),2),
        "DMS":dms, "Sector":SECTOR_NAMES.get(SECTOR_MAP.get(ticker,""),"—"),
        "win_prob":win_prob,"n_samp":n_samp,"l1":l1,
        "regime":regime,"exp_ret":exp_ret,"l2":l2,
        "z":z,"l3":l3,"kelly":kelly,"l4":l4,
        "p25":p25,"p50":p50,"p75":p75,"ploss":ploss,"l5":l5,
        "all_pass":all_pass,"ev":p50,
        "stop":round(float(c.iloc[-1])-1.5*float(atr),2),
        "trail":round(float(c.iloc[-1])-2.0*float(atr),2),
        "close_s":c,"mc_outcomes":mc_oc,
        "vix_now":vix_now,"spy_now":spy_now,"spy_200":spy_200,
    }

# ── LOAD WATCHLIST DATA ───────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def run_watchlist(wl_key, params_key):
    all_tix = tuple(set(list(wl_key)+list(SECTOR_MAP.values())+["SPY","^VIX","^TNX","^FVX"]))
    raw     = fetch_data(all_tix)
    vix_s   = get_col(raw,"^VIX","Close"); spy_s=get_col(raw,"SPY","Close")
    t10     = get_col(raw,"^TNX","Close"); t5   =get_col(raw,"^FVX","Close")
    out = []
    for ticker in wl_key:
        c  = get_col(raw,ticker,"Close"); v=get_col(raw,ticker,"Volume")
        sc = get_col(raw,SECTOR_MAP.get(ticker,"SPY"),"Close")
        if len(c)<60: continue
        try: out.append(evaluate(ticker,c,v,vix_s,spy_s,sc,t10,t5))
        except: continue
    return out

with st.spinner("Running statistical model..."):
    params_key = str((MIN_WIN_PROB,MIN_Z,MAX_Z,KELLY_FRAC,MC_N,MAX_POS,P25_FLOOR))
    results = run_watchlist(tuple(WATCHLIST), params_key)

if not results:
    st.error("No data returned. Check watchlist."); st.stop()

trades = [r for r in results if r["all_pass"]]
r0     = results[0]

# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-bar">
    <h2 style="color:white;margin:0">📊 Statistical Equity Model</h2>
    <p style="color:#90caf9;margin:4px 0 0">
    5-layer decision engine — only trade when the odds are provably in your favour</p>
</div>""", unsafe_allow_html=True)

regime_color = "#00e676" if r0["l2"] else "#ff1744"
spy_diff     = ((r0["spy_now"]/r0["spy_200"])-1)*100
st.markdown(
    f'<div style="background:#1c2030;border-left:4px solid {regime_color};'
    f'padding:10px 16px;border-radius:6px;margin-bottom:16px;">'
    f'<span style="color:{regime_color};font-weight:700;font-size:15px">'
    f'{"✅" if r0["l2"] else "❌"} Regime: {r0["regime"]}</span>'
    f'<span style="color:#888;font-size:13px;margin-left:16px">'
    f'VIX {r0["vix_now"]:.1f} &nbsp;|&nbsp; SPY {spy_diff:+.1f}% vs 200MA'
    f' &nbsp;|&nbsp; Expected monthly: {r0["exp_ret"]:+.1f}%</span></div>',
    unsafe_allow_html=True)

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Stocks analysed", len(results))
k2.metric("Trade signals today", len(trades))
k3.metric("Avg win probability", f"{np.mean([r['win_prob'] for r in results])*100:.1f}%")
k4.metric("Avg Z-score",         f"{np.mean([r['z'] for r in results]):.2f}")
k5.metric("Avg expected value",  f"${np.mean([r['ev'] for r in results]):,.0f}")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────

tab0,tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "🔭 Screener","🎯 Decisions","🔬 Statistical Detail",
    "📈 Charts","🎲 Monte Carlo","🔁 Backtest","📂 Positions"])

# ── TAB 0: SCREENER ───────────────────────────────────────────────────────────

with tab0:
    st.subheader("🔭 Universe Screener")
    st.caption("Scans S&P 100 (or a custom list) through pre-filters then the full "
               "5-layer statistical model. Surfaces only the highest-probability setups.")

    sc1,sc2,sc3 = st.columns(3)
    with sc1:
        universe_choice = st.selectbox("Universe", [
        "FTSE 100","S&P 500",
        "FTSE — Financials","FTSE — Healthcare","FTSE — Energy",
        "FTSE — Technology","FTSE — Consumer",
        "S&P — Technology (XLK)","S&P — Financials (XLF)",
        "S&P — Healthcare (XLV)","S&P — Energy (XLE)",
        "S&P — Industrials (XLI)","S&P — Consumer Disc. (XLY)",
        "Custom list"
    ])
    with sc2:
        min_vol_m = st.slider("Min avg volume (M/day)", 0.5, 5.0, 1.0, 0.5)
    with sc3:
        screen_top_n = st.slider("Return top N stocks", 3, 20, 10)

    if universe_choice == "Custom list":
        custom_input  = st.text_input("Tickers (comma-separated)",
                                      value="AAPL,MSFT,NVDA,TSLA,AMD,META,GOOGL,AMZN")
        screen_univ   = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    elif universe_choice == "S&P 500":
        screen_univ   = list(dict.fromkeys(SP500))  # deduplicated
    else:
        etf           = universe_choice.split("(")[1].replace(")","")
        screen_univ   = [t for t,s in SECTOR_MAP.items() if s==etf]

    st.info(f"Universe: **{len(list(dict.fromkeys(screen_univ)))} stocks** "
            f"{'🇬🇧 FTSE' if is_ftse else '🇺🇸 S&P'} "
            f"→ pre-filter → 5-layer model → top {screen_top_n}")

    if st.button("🔭 Run Screener", type="primary", use_container_width=True):
        prog = st.progress(0, text="Fetching data...")

        @st.cache_data(ttl=600)
        def fetch_screen(tickers_key):
            all_tix = list(set(list(tickers_key)+list(set(SECTOR_MAP.values()))+
                               ["SPY","^VIX","^TNX","^FVX"]))
            end=datetime.today(); start=end-timedelta(days=420)
            return yf.download(all_tix,start=start,end=end,
                               auto_adjust=True,progress=False,group_by="ticker")

        raw_sc = fetch_screen(tuple(screen_univ))
        vix_sc = get_col(raw_sc,"^VIX","Close"); spy_sc=get_col(raw_sc,"SPY","Close")
        t10_sc = get_col(raw_sc,"^TNX","Close"); t5_sc =get_col(raw_sc,"^FVX","Close")

        prog.progress(20, text="Pre-filtering...")

        regime_sc, tradeable_sc, *_ = classify_regime(vix_sc, spy_sc)

        pf_passed=[]; pf_reasons={}
        for ticker in screen_univ:
            c = get_col(raw_sc,ticker,"Close"); v=get_col(raw_sc,ticker,"Volume")
            if len(c)<55 or len(v)<20:
                pf_reasons[ticker]="Insufficient data"; continue
            price=c.iloc[-1]; avg_vol=v.rolling(20).mean().iloc[-1]
            ma50=c.rolling(50).mean().iloc[-1]; ret20=(c.iloc[-1]/c.iloc[-20])-1
            if price<10:
                pf_reasons[ticker]="Price<$10"; continue
            if avg_vol<min_vol_m*1e6:
                pf_reasons[ticker]="Low volume"; continue
            if c.iloc[-1]<ma50:
                pf_reasons[ticker]="Below 50MA"; continue
            if ret20<=0:
                pf_reasons[ticker]="Negative momentum"; continue
            pf_passed.append(ticker)

        prog.progress(40, text=f"{len(pf_passed)} passed pre-filter. Running model...")

        sc_results=[]
        for i,ticker in enumerate(pf_passed):
            try:
                c  = get_col(raw_sc,ticker,"Close"); v=get_col(raw_sc,ticker,"Volume")
                sc = get_col(raw_sc,
                            (FTSE_SECTOR_MAP if is_ftse else SECTOR_MAP).get(ticker,"SPY"),
                            "Close")
                if len(c)<60: continue
                dms      = compute_dms(c,v,vix_sc,spy_sc,sc,t10_sc,t5_sc)
                wp,n_s   = bayesian_win_prob(c,dms)
                z        = zscore_entry(c)
                kel      = kelly_size(wp)
                p25,p50,p75,ploss,_ = monte_carlo(c,kel,n=500) if kel>0 else (0,0,0,1.0,None)
                _rg,trd,*_ = classify_regime(vix_sc,spy_sc)
                l1=wp>=MIN_WIN_PROB; l2=trd; l3=MIN_Z<=z<=MAX_Z
                l4=kel>0.01;         l5=p25>P25_FLOOR
                atr=(c.rolling(2).max()-c.rolling(2).min()).rolling(14).mean().iloc[-1]
                sc_results.append({
                    "Ticker":ticker,"Price":round(float(c.iloc[-1]),2),
                    "DMS":dms,"Win Prob %":round(wp*100,1),
                    "Z-Score":z,"Kelly %":round(kel*100,1),
                    "P25 $":p25,"EV $":p50,"P75 $":p75,
                    "P(loss) %":round(ploss*100,1),
                    "All pass":l1 and l2 and l3 and l4 and l5,
                    "L1":("✅" if l1 else "❌"),"L2":("✅" if l2 else "❌"),
                    "L3":("✅" if l3 else "❌"),"L4":("✅" if l4 else "❌"),
                    "L5":("✅" if l5 else "❌"),
                    "Stop":round(float(c.iloc[-1])-1.5*float(atr),2),
                    "Sector":SECTOR_NAMES.get(SECTOR_MAP.get(ticker,""),"—"),
                })
            except: pass
            prog.progress(40+int(55*(i+1)/max(len(pf_passed),1)),
                          text=f"Scoring {ticker}...")

        prog.progress(100, text="Done.")

        if not sc_results:
            st.warning("No results generated.")
        else:
            sc_df   = pd.DataFrame(sc_results).sort_values("EV $", ascending=False)
            winners = sc_df[sc_df["All pass"]]
            near    = sc_df[~sc_df["All pass"]].head(10)

            s1,s2,s3,s4 = st.columns(4)
            s1.metric("Universe scanned",  len(screen_univ))
            s2.metric("Passed pre-filter", len(pf_passed))
            s3.metric("Pass all 5 layers", len(winners))
            s4.metric("Regime", regime_sc,
                      delta="Tradeable ✅" if tradeable_sc else "Bear ❌",
                      delta_color="normal" if tradeable_sc else "inverse")

            st.markdown("---")

            if len(winners):
                st.success(f"✅ **{len(winners)} stock(s) pass all 5 layers:** "
                           f"{', '.join(winners['Ticker'].tolist())}")
                st.markdown("#### Top trade candidates")

                def chighlight(v):
                    return ("color:#00e676;font-weight:bold" if v=="✅"
                            else "color:#ff1744;font-weight:bold")
                def cev(v):
                    return f"color:{'#00e676' if v>0 else '#ff1744'};font-weight:bold"

                dcols = ["Ticker","Price","DMS","Win Prob %","Z-Score",
                         "Kelly %","P25 $","EV $","Stop","Sector","L1","L2","L3","L4","L5"]
                st.dataframe(
                    winners[dcols].head(screen_top_n).style
                        .map(chighlight, subset=["L1","L2","L3","L4","L5"])
                        .map(cev,        subset=["EV $"]),
                    use_container_width=True,
                    height=min(80+len(winners)*40, 420))

                fig_ev = px.bar(
                    winners.head(screen_top_n).sort_values("EV $"),
                    x="EV $", y="Ticker", orientation="h",
                    color="Win Prob %", color_continuous_scale="RdYlGn",
                    range_color=[45,75],
                    title="Expected value by stock",
                    height=max(250, len(winners)*50))
                fig_ev.update_layout(paper_bgcolor="#0e1117",
                                     plot_bgcolor="#1c2030",font_color="white")
                st.plotly_chart(fig_ev, use_container_width=True)
            else:
                st.warning("❌ No stocks pass all 5 layers today.")

            st.markdown("#### Near misses (1–2 layers failing)")
            if len(near):
                st.dataframe(
                    near[["Ticker","Price","DMS","Win Prob %","Z-Score",
                           "EV $","L1","L2","L3","L4","L5","Sector"]],
                    use_container_width=True, height=300)
            else:
                st.info("No near misses.")

            st.download_button(
                "⬇️ Download screener results",
                sc_df.to_csv(index=False),
                f"screener_{datetime.today().strftime('%Y%m%d')}.csv",
                "text/csv")

# ── TAB 1: DECISIONS ──────────────────────────────────────────────────────────

with tab1:
    if trades:
        st.success(f"✅ **Trade today:** {', '.join([r['Ticker'] for r in trades])}")
    else:
        st.warning("❌ No stocks pass all 5 layers today. Stand aside.")

    st.markdown("#### Decision breakdown")
    for r in sorted(results, key=lambda x: -x["ev"]):
        border = "#00e676" if r["all_pass"] else "#ff1744"
        c0,c1,c2,c3,c4,c5,c6 = st.columns([1.2,1.5,1.5,1,1.5,1.2,1])
        with c0:
            st.markdown(
                f'<div style="background:#1c2030;border-left:3px solid {border};'
                f'padding:8px 12px;border-radius:6px;">'
                f'<div style="color:white;font-weight:700;font-size:16px">{r["Ticker"]}</div>'
                f'<div style="color:#888;font-size:11px">${r["Price"]:.2f}</div></div>',
                unsafe_allow_html=True)
        with c1:
            col = "#00e676" if r["l1"] else "#ff1744"
            st.markdown(
                f'<div style="background:#1c2030;border-radius:6px;padding:8px 12px;">'
                f'<div style="color:#90caf9;font-size:10px">L1 Win probability</div>'
                f'<div style="color:{col};font-size:18px;font-weight:700">'
                f'{r["win_prob"]*100:.1f}% {"✅" if r["l1"] else "❌"}</div>'
                f'<div style="color:#555;font-size:10px">n={r["n_samp"]}</div></div>',
                unsafe_allow_html=True)
        with c2:
            col = "#00e676" if r["l2"] else "#ff1744"
            st.markdown(
                f'<div style="background:#1c2030;border-radius:6px;padding:8px 12px;">'
                f'<div style="color:#90caf9;font-size:10px">L2 Regime</div>'
                f'<div style="color:{col};font-size:14px;font-weight:700">'
                f'{r["regime"]} {"✅" if r["l2"] else "❌"}</div>'
                f'<div style="color:#555;font-size:10px">{r["exp_ret"]:+.1f}%/mo</div></div>',
                unsafe_allow_html=True)
        with c3:
            col = "#00e676" if r["l3"] else "#ff1744"
            st.markdown(
                f'<div style="background:#1c2030;border-radius:6px;padding:8px 12px;">'
                f'<div style="color:#90caf9;font-size:10px">L3 Z-score</div>'
                f'<div style="color:{col};font-size:18px;font-weight:700">'
                f'{r["z"]:.2f} {"✅" if r["l3"] else "❌"}</div>'
                f'<div style="color:#555;font-size:10px">0.5–2.5 zone</div></div>',
                unsafe_allow_html=True)
        with c4:
            col = "#00e676" if r["l4"] else "#ff1744"
            st.markdown(
                f'<div style="background:#1c2030;border-radius:6px;padding:8px 12px;">'
                f'<div style="color:#90caf9;font-size:10px">L4 Kelly size</div>'
                f'<div style="color:{col};font-size:18px;font-weight:700">'
                f'{r["kelly"]*100:.1f}% {"✅" if r["l4"] else "❌"}</div>'
                f'<div style="color:#555;font-size:10px">Half-Kelly</div></div>',
                unsafe_allow_html=True)
        with c5:
            col = "#00e676" if r["l5"] else "#ff1744"
            st.markdown(
                f'<div style="background:#1c2030;border-radius:6px;padding:8px 12px;">'
                f'<div style="color:#90caf9;font-size:10px">L5 Monte Carlo P25</div>'
                f'<div style="color:{col};font-size:16px;font-weight:700">'
                f'${r["p25"]:,.0f} {"✅" if r["l5"] else "❌"}</div>'
                f'<div style="color:#555;font-size:10px">P(loss) {r["ploss"]*100:.0f}%</div></div>',
                unsafe_allow_html=True)
        with c6:
            bg  = "#00e676" if r["all_pass"] else "#333"
            txt = "#003300" if r["all_pass"] else "#888"
            st.markdown(
                f'<div style="background:{bg};border-radius:6px;padding:8px 12px;text-align:center;">'
                f'<div style="color:{txt};font-size:12px;font-weight:700">'
                f'{"TRADE" if r["all_pass"] else "SKIP"}</div>'
                f'<div style="color:{txt};font-size:11px">EV ${r["ev"]:+,.0f}</div></div>',
                unsafe_allow_html=True)
        st.markdown("")

# ── TAB 2: STATISTICAL DETAIL ─────────────────────────────────────────────────

with tab2:
    sel = st.selectbox("Select stock for deep dive", [r["Ticker"] for r in results])
    r   = next(x for x in results if x["Ticker"]==sel)

    st.markdown(f"### {sel} — Statistical breakdown")
    col1,col2 = st.columns(2)

    with col1:
        st.markdown("#### Layer 1 — Bayesian win probability")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=r["win_prob"]*100,
            number={"suffix":"%"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":"#00e676" if r["l1"] else "#ff1744"},
                "steps":[{"range":[0,45],"color":"#b71c1c"},
                          {"range":[45,55],"color":"#f57f17"},
                          {"range":[55,70],"color":"#2e7d32"},
                          {"range":[70,100],"color":"#1b5e20"}],
                "threshold":{"line":{"color":"white","width":3},
                             "value":MIN_WIN_PROB*100}},
            title={"text":f"Win prob (need ≥{MIN_WIN_PROB*100:.0f}%)",
                   "font":{"color":"white"}}))
        fig_g.update_layout(paper_bgcolor="#0e1117",font_color="white",height=260)
        st.plotly_chart(fig_g, use_container_width=True)
        st.caption(f"Based on {r['n_samp']} historical observations at DMS ±10 of {r['DMS']}")

    with col2:
        st.markdown("#### Layer 3 — Z-score entry timing")
        x  = np.linspace(-4,4,200)
        y  = np.exp(-0.5*x**2)/np.sqrt(2*np.pi)
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=x,y=y,fill="tozeroy",
                                   fillcolor="rgba(33,150,243,0.15)",
                                   line=dict(color="#2196F3",width=1.5)))
        mask = (x>=MIN_Z)&(x<=MAX_Z)
        fig_z.add_trace(go.Scatter(
            x=np.concatenate([x[mask],[x[mask][-1],x[mask][0]]]),
            y=np.concatenate([y[mask],[0,0]]),
            fill="toself",fillcolor="rgba(0,230,118,0.25)",
            line=dict(width=0),name="Ideal zone"))
        fig_z.add_vline(x=r["z"],
                        line_color="#00e676" if r["l3"] else "#ff1744",
                        line_width=2,
                        annotation_text=f"Today: {r['z']:.2f}",
                        annotation_font_color="#00e676" if r["l3"] else "#ff1744")
        fig_z.update_layout(
            paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
            font_color="white",height=260,showlegend=False,
            title=dict(text=f"Z-score (ideal: {MIN_Z}–{MAX_Z})",font=dict(color="white")),
            xaxis=dict(title="Standard deviations",color="white"),
            yaxis=dict(visible=False))
        st.plotly_chart(fig_z, use_container_width=True)

    st.markdown("#### Layer 4 — Kelly criterion sizing")
    kc1,kc2,kc3 = st.columns(3)
    kc1.metric("Win probability",    f"{r['win_prob']*100:.1f}%")
    kc2.metric("Full Kelly",         f"{r['kelly']/KELLY_FRAC*100:.1f}%")
    kc3.metric("Half-Kelly (used)",  f"{r['kelly']*100:.1f}%",
               delta=f"${100000*r['kelly']:,.0f} on $100k")

    st.markdown("#### Layer 5 — Monte Carlo distribution")
    mc_oc = r["mc_outcomes"]
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(x=mc_oc,nbinsx=60,
                                  marker_color="#2196F3",opacity=0.75))
    for pct,col,lbl in [(25,"#ff6e40","P25"),(50,"#ffd740","P50"),(75,"#00e676","P75")]:
        v2 = np.percentile(mc_oc,pct)
        fig_mc.add_vline(x=v2,line_dash="dash",line_color=col,
                         annotation_text=f"{lbl}: ${v2:,.0f}",
                         annotation_font_color=col)
    fig_mc.add_vline(x=0,line_color="#555",line_width=1)
    fig_mc.update_layout(
        paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",font_color="white",height=320,
        xaxis_title="10-day P&L ($)",yaxis_title="Frequency",
        title=dict(text=(f"{sel} — P(loss): {r['ploss']*100:.1f}%  |  "
                         f"Kelly: {r['kelly']*100:.1f}%  |  "
                         f"P25 floor: ${P25_FLOOR:,}  {'✅' if r['l5'] else '❌'}"),
                   font=dict(color="white")))
    st.plotly_chart(fig_mc, use_container_width=True)

    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.metric("P25 (bad scenario)",  f"${r['p25']:,.0f}",
               delta="Pass ✅" if r["l5"] else "Fail ❌",
               delta_color="normal" if r["l5"] else "inverse")
    mc2.metric("P50 (median)",        f"${r['p50']:,.0f}")
    mc3.metric("P75 (good scenario)", f"${r['p75']:,.0f}")
    mc4.metric("Probability of loss", f"{r['ploss']*100:.1f}%")

# ── TAB 3: PRICE CHARTS ───────────────────────────────────────────────────────

with tab3:
    sel2 = st.selectbox("Select stock", [r["Ticker"] for r in results], key="ch2")
    r2   = next(x for x in results if x["Ticker"]==sel2)
    cls  = r2["close_s"]

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=cls.index,y=cls.values,name="Price",
                               line=dict(color="#2196F3",width=2)))
    fig_p.add_trace(go.Scatter(x=cls.index,y=cls.rolling(20).mean(),name="20MA",
                               line=dict(color="#ffd740",width=1.5,dash="dash")))
    fig_p.add_trace(go.Scatter(x=cls.index,y=cls.rolling(50).mean(),name="50MA",
                               line=dict(color="#ff6e40",width=1.5,dash="dot")))
    fig_p.add_trace(go.Scatter(x=cls.index,y=cls.rolling(200).mean(),name="200MA",
                               line=dict(color="#9c27b0",width=1,dash="dot")))
    fig_p.add_hline(y=r2["stop"],line_dash="dash",line_color="#ff1744",
                    annotation_text=f"Stop {r2['stop']}",annotation_font_color="#ff1744")
    fig_p.add_hline(y=r2["trail"],line_dash="dot",line_color="#ff6e40",
                    annotation_text=f"Trail {r2['trail']}",annotation_font_color="#ff6e40")
    fig_p.update_layout(
        title=(f"{sel2}  |  DMS {r2['DMS']}  |  WinP {r2['win_prob']*100:.1f}%  |  "
               f"Kelly {r2['kelly']*100:.1f}%  |  {'✅ TRADE' if r2['all_pass'] else '❌ SKIP'}"),
        paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",font_color="white",height=460)
    st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("#### Rolling 20-day win probability")
    ret_s   = cls.pct_change().dropna()
    rm      = ret_s.rolling(20).mean(); rs2 = ret_s.rolling(20).std()
    proxy   = ((rm/rs2.replace(0,np.nan)).clip(-3,3)*16.67+50).dropna()
    fwd     = cls.pct_change(10).shift(-10)
    aln     = pd.concat([proxy,fwd],axis=1).dropna(); aln.columns=["d","f"]
    roll_wp = aln["f"].gt(0).rolling(30).mean()
    fig_wp  = go.Figure()
    fig_wp.add_trace(go.Scatter(x=roll_wp.index,y=roll_wp*100,
                                fill="tozeroy",fillcolor="rgba(33,150,243,0.15)",
                                line=dict(color="#2196F3",width=1.5)))
    fig_wp.add_hline(y=MIN_WIN_PROB*100,line_dash="dash",line_color="#00e676",
                     annotation_text=f"Threshold {MIN_WIN_PROB*100:.0f}%")
    fig_wp.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                         font_color="white",height=220,
                         yaxis=dict(range=[0,100],title="Win prob %"),xaxis_title="Date")
    st.plotly_chart(fig_wp, use_container_width=True)

# ── TAB 4: MONTE CARLO EXPLORER ───────────────────────────────────────────────

with tab4:
    st.subheader("🎲 Monte Carlo Explorer")
    mc_ticker = st.selectbox("Stock",[r["Ticker"] for r in results],key="mc_t")
    mc_r      = next(x for x in results if x["Ticker"]==mc_ticker)
    mc_cls    = mc_r["close_s"]

    mc1c,mc2c,mc3c = st.columns(3)
    with mc1c: mc_capital = st.number_input("Portfolio size ($)",value=100000,step=10000)
    with mc2c: mc_pos_pct = st.slider("Position size %",1,25,int(mc_r["kelly"]*100))
    with mc3c: mc_horizon = st.slider("Holding period (days)",5,30,10)

    if st.button("▶️ Run simulation", type="primary"):
        ret_mc  = mc_cls.pct_change().dropna().values
        entry   = mc_cls.iloc[-1]; shares=(mc_capital*(mc_pos_pct/100))/entry
        sims    = np.array([(entry*np.prod(1+np.random.choice(ret_mc,size=mc_horizon,
                             replace=True))-entry)*shares for _ in range(MC_N)])
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("P10", f"${np.percentile(sims,10):,.0f}")
        s2.metric("P25", f"${np.percentile(sims,25):,.0f}")
        s3.metric("P50", f"${np.percentile(sims,50):,.0f}")
        s4.metric("P75", f"${np.percentile(sims,75):,.0f}")
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Histogram(x=sims,nbinsx=80,
                                       marker_color="#2196F3",opacity=0.75))
        for pct,col,lbl in [(10,"#ff1744","P10"),(25,"#ff6e40","P25"),
                             (50,"#ffd740","P50"),(75,"#00e676","P75")]:
            v3=np.percentile(sims,pct)
            fig_sim.add_vline(x=v3,line_dash="dash",line_color=col,
                              annotation_text=f"{lbl}: ${v3:,.0f}",
                              annotation_font_color=col)
        fig_sim.add_vline(x=0,line_color="#555",line_width=1.5)
        fig_sim.update_layout(
            paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",font_color="white",height=380,
            title=dict(text=(f"{mc_ticker} — {MC_N:,} simulations  |  "
                             f"Position {mc_pos_pct}%  |  {mc_horizon}d  |  "
                             f"P(loss): {(sims<0).mean()*100:.1f}%"),
                       font=dict(color="white")),
            xaxis_title="P&L ($)",yaxis_title="Frequency")
        st.plotly_chart(fig_sim, use_container_width=True)
        kelly_rec = kelly_size(mc_r["win_prob"])*100
        if mc_pos_pct > kelly_rec*1.5:
            st.warning(f"⚠️ {mc_pos_pct}% is above Kelly recommendation ({kelly_rec:.1f}%). "
                       f"Higher ruin risk.")

# ── TAB 5: BACKTEST ───────────────────────────────────────────────────────────

with tab5:
    st.subheader("🔁 Backtest — statistical model")
    bc1,bc2,bc3 = st.columns(3)
    with bc1: bt_tick  = st.text_input("Ticker (any stock)", value="AAPL").upper()
    with bc2:
        bt_period = st.selectbox("Period",
            ["5 years","3 years","2 years","1 year","Custom"], index=0)
    with bc3: bt_cap = st.number_input("Capital ($)", value=100000, step=10000)

    if bt_period == "Custom":
        cc1, cc2 = st.columns(2)
        with cc1: bt_start = st.date_input("Start date",
                             value=datetime.today()-timedelta(days=5*365))
        with cc2: bt_end   = st.date_input("End date", value=datetime.today())
    else:
        years    = int(bt_period.split()[0])
        bt_start = datetime.today() - timedelta(days=years*365)
        bt_end   = datetime.today()

    st.info(f"Backtest: **{bt_tick}** from "
            f"**{bt_start.strftime('%d %b %Y') if hasattr(bt_start,'strftime') else bt_start}**"
            f" → **{bt_end.strftime('%d %b %Y') if hasattr(bt_end,'strftime') else bt_end}**")

    @st.cache_data(ttl=600)
    def backtest_stat(ticker,start_str,end_str,capital,params_k):
        all_tix=list(set([ticker,SECTOR_MAP.get(ticker,"SPY"),
                          "SPY","^VIX","^TNX","^FVX"]))
        end_dt = datetime.today()
        raw=yf.download(all_tix,start=start_str,end=end_dt.strftime("%Y-%m-%d"),
                        auto_adjust=True,progress=False,group_by="ticker")
        c=get_col(raw,ticker,"Close"); v=get_col(raw,ticker,"Volume")
        vix=get_col(raw,"^VIX","Close"); spy=get_col(raw,"SPY","Close")
        sc=get_col(raw,SECTOR_MAP.get(ticker,"SPY"),"Close")
        t10=get_col(raw,"^TNX","Close"); t5=get_col(raw,"^FVX","Close")
        if len(c)<60: return None,None,None,None
        idx=c.index
        vix=vix.reindex(idx,method="ffill"); spy=spy.reindex(idx,method="ffill")
        sc=sc.reindex(idx,method="ffill"); t10=t10.reindex(idx,method="ffill")
        t5=t5.reindex(idx,method="ffill")
        atr_s=(c.rolling(2).max()-c.rolling(2).min()).rolling(14).mean()
        cap=float(capital); pos=0; entry_px=0.0; stop_px=0.0; highest=0.0
        trades=[]; equity=[]; wp_list=[]
        for i in range(len(c)):
            price=c.iloc[i]; date=idx[i]
            atr=atr_s.iloc[i] if not pd.isna(atr_s.iloc[i]) else 0
            ci=c.iloc[:i+1]; vi=v.iloc[:i+1]
            vxi=vix.iloc[:i+1]; spi=spy.iloc[:i+1]; sci=sc.iloc[:i+1]
            t10i=t10.iloc[:i+1]; t5i=t5.iloc[:i+1]
            if len(ci)<60:
                equity.append(cap+pos*price); wp_list.append(0.5); continue
            try:
                dms=compute_dms(ci,vi,vxi,spi,sci,t10i,t5i)
                wp,_=bayesian_win_prob(ci,dms)
                z=zscore_entry(ci)
                _rg,trd,*_=classify_regime(vxi,spi)
                kel=kelly_size(wp)
                wp_list.append(wp)
            except:
                equity.append(cap+pos*price); wp_list.append(0.5); continue
            if pos>0:
                highest=max(price,highest)
                stop_px=max(stop_px,highest-2.0*atr)
            if pos>0 and price<=stop_px:
                pnl=(price-entry_px)*pos; cap+=pos*price
                trades.append({"Date":date,"Action":"STOP","Price":round(price,2),
                                "Shares":pos,"PnL":round(pnl,2)}); pos=0
            all_pass=(wp>=MIN_WIN_PROB and trd and MIN_Z<=z<=MAX_Z and kel>0.01)
            if all_pass and pos==0:
                shares=int(cap*kel/price)
                if shares>0:
                    cap-=shares*price; pos=shares; entry_px=price
                    highest=price; stop_px=price-1.5*atr
                    trades.append({"Date":date,"Action":"BUY","Price":round(price,2),
                                   "Shares":shares,"PnL":0})
            elif wp<0.45 and pos>0:
                pnl=(price-entry_px)*pos; cap+=pos*price
                trades.append({"Date":date,"Action":"SELL","Price":round(price,2),
                                "Shares":pos,"PnL":round(pnl,2)}); pos=0
            equity.append(cap+pos*price)
        if pos>0:
            price=c.iloc[-1]; pnl=(price-entry_px)*pos; cap+=pos*price
            trades.append({"Date":idx[-1],"Action":"CLOSE","Price":round(price,2),
                            "Shares":pos,"PnL":round(pnl,2)})
        return (pd.DataFrame(trades) if trades else
                pd.DataFrame(columns=["Date","Action","Price","Shares","PnL"]),
                pd.Series(equity,index=idx),
                pd.Series(wp_list,index=idx), c)

    if st.button("▶️ Run backtest", type="primary"):
        with st.spinner(f"Backtesting {bt_tick} over {bt_period}..."):
            pk  = str((MIN_WIN_PROB,MIN_Z,MAX_Z,KELLY_FRAC,MAX_POS))
            bts = str(bt_start.date() if hasattr(bt_start,"date") else bt_start)
            bte = str(bt_end.date()   if hasattr(bt_end,  "date") else bt_end)
            tdf,eq_s,wp_s,cls_bt = backtest_stat(bt_tick,bts,bte,bt_cap,pk)

        if eq_s is None:
            st.error("Not enough data.")
        else:
            tr=(eq_s.iloc[-1]-bt_cap)/bt_cap*100
            bh=(cls_bt.iloc[-1]-cls_bt.iloc[0])/cls_bt.iloc[0]*100
            dr=eq_s.pct_change().dropna()
            sh=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
            dd=((eq_s-eq_s.cummax())/eq_s.cummax()*100).min()
            cl=tdf[tdf["Action"].isin(["SELL","STOP","CLOSE"])]
            n=len(cl); wi=cl[cl["PnL"]>0]; lo=cl[cl["PnL"]<=0]
            hr=len(wi)/n*100 if n>0 else 0
            aw=wi["PnL"].mean() if len(wi)>0 else 0
            al=lo["PnL"].mean() if len(lo)>0 else 0
            pf=abs(aw/al) if al!=0 else 0

            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Total return",  f"{tr:.1f}%", delta=f"{tr-bh:+.1f}% vs B&H")
            m2.metric("Buy & hold",    f"{bh:.1f}%")
            m3.metric("Sharpe",        round(sh,2),
                      delta="Good" if sh>=1 else "Weak",
                      delta_color="normal" if sh>=1 else "inverse")
            m4.metric("Max drawdown",  f"{dd:.1f}%")
            m5.metric("Win rate",      f"{hr:.1f}%")
            m6.metric("Final value",   f"${eq_s.iloc[-1]:,.0f}")

            st.markdown("---")

            fig_bt=go.Figure()
            fig_bt.add_trace(go.Scatter(x=cls_bt.index,y=cls_bt.values,name="Price",
                                        line=dict(color="#2196F3",width=1.5)))
            fig_bt.add_trace(go.Scatter(x=cls_bt.index,y=cls_bt.rolling(50).mean(),
                                        name="50MA",line=dict(color="#ff6e40",width=1,dash="dot")))
            if not tdf.empty:
                for act,sym,col in [("BUY","triangle-up","#00e676"),
                                     ("SELL","triangle-down","#ffd740"),
                                     ("STOP","x","#ff1744")]:
                    t=tdf[tdf["Action"]==act]
                    if len(t):
                        fig_bt.add_trace(go.Scatter(x=t["Date"],y=t["Price"],
                            mode="markers",marker=dict(symbol=sym,size=12,color=col),name=act))
            fig_bt.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                                 font_color="white",height=380,
                                 title=f"{bt_tick} — Statistical model signals")
            st.plotly_chart(fig_bt, use_container_width=True)

            bh_eq=(cls_bt/cls_bt.iloc[0])*bt_cap
            fig_eq=go.Figure()
            fig_eq.add_trace(go.Scatter(x=eq_s.index,y=eq_s.values,name="Model",
                                        line=dict(color="#00e676",width=2)))
            fig_eq.add_trace(go.Scatter(x=bh_eq.index,y=bh_eq.values,name="Buy & hold",
                                        line=dict(color="#90caf9",width=2,dash="dash")))
            fig_eq.add_hline(y=bt_cap,line_dash="dot",line_color="#555")
            fig_eq.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                                 font_color="white",height=300,yaxis_title="Value ($)")
            st.plotly_chart(fig_eq, use_container_width=True)

            fig_wp2=go.Figure()
            fig_wp2.add_trace(go.Scatter(x=wp_s.index,y=wp_s*100,
                                         fill="tozeroy",
                                         fillcolor="rgba(33,150,243,0.1)",
                                         line=dict(color="#2196F3",width=1)))
            fig_wp2.add_hline(y=MIN_WIN_PROB*100,line_dash="dash",line_color="#00e676",
                              annotation_text=f"Threshold {MIN_WIN_PROB*100:.0f}%")
            fig_wp2.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                                  font_color="white",height=200,
                                  yaxis=dict(range=[0,100],title="Win prob %"),
                                  title="Rolling Bayesian win probability")
            st.plotly_chart(fig_wp2, use_container_width=True)

            if not tdf.empty:
                st.markdown("#### Trade log")
                def ca(val):
                    c={"BUY":"#00e676","SELL":"#ffd740","STOP":"#ff1744",
                       "CLOSE":"#90caf9"}.get(val,"white")
                    return f"color:{c};font-weight:bold"
                def cp(val):
                    return f"color:{'#00e676' if val>0 else '#ff1744' if val<0 else 'white'}"
                st.dataframe(tdf.style.map(ca,subset=["Action"]).map(cp,subset=["PnL"]),
                             use_container_width=True,height=260)
                st.download_button("⬇️ Download trades",tdf.to_csv(index=False),
                                   f"trades_{bt_tick}.csv","text/csv")

# ── TAB 6: POSITION TRACKER ───────────────────────────────────────────────────

with tab6:
    st.subheader("📂 Position Tracker")
    st.caption("Log your open trades. The model monitors all exit conditions daily "
               "and tells you when to act.")

    # ── Initialise session state ──
    if "positions" not in st.session_state:
        st.session_state.positions = []

    # ── Add new position ──
    st.markdown("#### Add a position")
    p1,p2,p3,p4,p5 = st.columns(5)
    with p1: pos_ticker  = st.text_input("Ticker",  value="AAPL").upper()
    with p2: pos_entry   = st.number_input("Entry price ($)", value=100.0, step=0.01)
    with p3: pos_shares  = st.number_input("Shares", value=10, step=1)
    with p4: pos_stop    = st.number_input("Stop-loss ($)", value=90.0, step=0.01)
    with p5: pos_date    = st.date_input("Entry date", value=datetime.today())

    pos_max_weeks = st.slider("Maximum hold (weeks)", 2, 16, 8)

    if st.button("➕ Add position", type="primary"):
        st.session_state.positions.append({
            "ticker":    pos_ticker,
            "entry":     pos_entry,
            "shares":    pos_shares,
            "stop":      pos_stop,
            "date":      str(pos_date),
            "max_weeks": pos_max_weeks,
            "trail_high":pos_entry,
        })
        st.success(f"Added {pos_ticker} — {pos_shares} shares at ${pos_entry:.2f}")

    if not st.session_state.positions:
        st.info("No open positions yet. Add one above.")
    else:
        st.markdown("---")
        st.markdown("#### Open positions — live monitoring")

        @st.cache_data(ttl=300)
        def fetch_position_data(tickers_key):
            end = datetime.today(); start = end - timedelta(days=420)
            all_tix = list(set(list(tickers_key) +
                               list(SECTOR_MAP.values()) +
                               ["SPY","^VIX","^TNX","^FVX"]))
            return yf.download(all_tix, start=start, end=end,
                               auto_adjust=True, progress=False, group_by="ticker")

        pos_tickers = tuple(set(p["ticker"] for p in st.session_state.positions))
        raw_pos     = fetch_position_data(pos_tickers)
        vix_pos     = get_col(raw_pos,"^VIX","Close")
        spy_pos     = get_col(raw_pos,"SPY","Close")
        t10_pos     = get_col(raw_pos,"^TNX","Close")
        t5_pos      = get_col(raw_pos,"^FVX","Close")
        regime_pos, tradeable_pos, *_ = classify_regime(vix_pos, spy_pos)

        to_remove = []

        for idx_p, pos in enumerate(st.session_state.positions):
            ticker    = pos["ticker"]
            c         = get_col(raw_pos, ticker, "Close")
            v         = get_col(raw_pos, ticker, "Volume")
            sc        = get_col(raw_pos, SECTOR_MAP.get(ticker,"SPY"), "Close")

            if len(c) < 60:
                st.warning(f"{ticker}: insufficient data."); continue

            # Current stats
            current_price = float(c.iloc[-1])
            entry_price   = pos["entry"]
            shares        = pos["shares"]
            stop          = pos["stop"]
            entry_date    = datetime.strptime(pos["date"], "%Y-%m-%d")
            days_held     = (datetime.today() - entry_date).days
            weeks_held    = days_held / 7
            max_weeks     = pos["max_weeks"]

            # P&L
            pnl_pct  = (current_price - entry_price) / entry_price * 100
            pnl_cash = (current_price - entry_price) * shares
            position_value = current_price * shares

            # ATR-based trailing stop
            atr = float((c.rolling(2).max()-c.rolling(2).min()).rolling(14).mean().iloc[-1])
            trail_stop = current_price - 2.0 * atr
            # Update to highest price seen
            trail_high = max(pos.get("trail_high", entry_price), current_price)
            st.session_state.positions[idx_p]["trail_high"] = trail_high
            trail_stop_actual = trail_high - 2.0 * atr
            breakeven_stop    = entry_price  # move stop here after week 2

            # Statistical signals
            try:
                dms      = compute_dms(c, v, vix_pos, spy_pos, sc, t10_pos, t5_pos)
                win_prob, _ = bayesian_win_prob(c, dms)
                z        = zscore_entry(c)
            except:
                dms=50; win_prob=0.5; z=0.0

            # Exit conditions
            exit_signals = []
            exit_urgent  = []

            if current_price <= stop:
                exit_urgent.append("🔴 STOP-LOSS HIT — exit immediately")
            if current_price <= trail_stop_actual:
                exit_urgent.append("🔴 TRAILING STOP HIT — exit immediately")
            if win_prob < 0.45:
                exit_signals.append(f"⚠️ Win probability dropped to {win_prob*100:.1f}% (below 45%)")
            if dms < 40:
                exit_signals.append(f"⚠️ DMS fell to {dms:.0f} (below 40)")
            if z > 3.0:
                exit_signals.append(f"⚠️ Z-score overbought at {z:.2f} — consider trimming")
            if not tradeable_pos:
                exit_signals.append(f"⚠️ Regime turned {regime_pos} — consider exiting")
            if weeks_held >= max_weeks:
                exit_signals.append(f"⚠️ Maximum hold of {max_weeks} weeks reached")

            # Profit milestones
            profit_flags = []
            if pnl_cash >= 3 * atr * shares:
                profit_flags.append(f"💰 +3× ATR gain — consider selling ⅓")
            if pnl_cash >= 2 * atr * shares:
                profit_flags.append(f"💰 +2× ATR gain — consider selling ⅓")
            if weeks_held >= 2 and pnl_pct > 0:
                profit_flags.append(f"✅ Week 2+ in profit — move stop to breakeven (${entry_price:.2f})")

            # Overall status
            if exit_urgent:
                status_color = "#ff1744"; status_label = "⛔ EXIT NOW"
            elif exit_signals:
                status_color = "#ff6e40"; status_label = "⚠️ REVIEW"
            elif profit_flags:
                status_color = "#ffd740"; status_label = "💰 MANAGE"
            else:
                status_color = "#00e676"; status_label = "✅ HOLD"

            # ── Card ──
            pnl_color = "#00e676" if pnl_cash >= 0 else "#ff1744"

            with st.container():
                st.markdown(
                    f'<div style="background:#1c2030;border-left:4px solid {status_color};'
                    f'border-radius:8px;padding:16px 20px;margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<div>'
                    f'<span style="color:white;font-size:20px;font-weight:700">{ticker}</span>'
                    f'<span style="color:#888;font-size:13px;margin-left:12px">'
                    f'{shares} shares · entered ${entry_price:.2f} on {pos["date"]}</span>'
                    f'</div>'
                    f'<div style="text-align:right;">'
                    f'<span style="color:{status_color};font-size:15px;font-weight:700">'
                    f'{status_label}</span>'
                    f'</div></div></div>',
                    unsafe_allow_html=True)

                # Metrics row
                m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
                m1.metric("Current price",  f"${current_price:.2f}")
                m2.metric("P&L",            f"${pnl_cash:+,.0f}",
                          delta=f"{pnl_pct:+.1f}%",
                          delta_color="normal" if pnl_cash>=0 else "inverse")
                m3.metric("Position value", f"${position_value:,.0f}")
                m4.metric("Days held",      f"{days_held}d",
                          delta=f"{weeks_held:.1f}/{max_weeks} wks")
                m5.metric("Win prob",       f"{win_prob*100:.1f}%",
                          delta="✅" if win_prob>=0.55 else "❌",
                          delta_color="normal" if win_prob>=0.55 else "inverse")
                m6.metric("DMS",            f"{dms:.0f}",
                          delta="✅" if dms>=40 else "❌",
                          delta_color="normal" if dms>=40 else "inverse")
                m7.metric("Trail stop",     f"${trail_stop_actual:.2f}")

                # Exit alerts
                if exit_urgent:
                    for msg in exit_urgent:
                        st.error(msg)
                if exit_signals:
                    for msg in exit_signals:
                        st.warning(msg)
                if profit_flags and not exit_urgent:
                    for msg in profit_flags:
                        st.info(msg)
                if not exit_urgent and not exit_signals and not profit_flags:
                    st.success(f"All conditions healthy — hold. "
                               f"Next review in {max(1, 7-days_held%7)} days.")

                # Mini price chart
                with st.expander(f"📈 {ticker} price chart"):
                    fig_pos = go.Figure()
                    fig_pos.add_trace(go.Scatter(
                        x=c.index, y=c.values, name="Price",
                        line=dict(color="#2196F3", width=1.5)))
                    fig_pos.add_trace(go.Scatter(
                        x=c.index, y=c.rolling(20).mean(), name="20MA",
                        line=dict(color="#ffd740", width=1, dash="dash")))
                    fig_pos.add_hline(y=entry_price, line_dash="dot",
                                      line_color="#90caf9",
                                      annotation_text=f"Entry ${entry_price:.2f}",
                                      annotation_font_color="#90caf9")
                    fig_pos.add_hline(y=stop, line_dash="dash",
                                      line_color="#ff1744",
                                      annotation_text=f"Stop ${stop:.2f}",
                                      annotation_font_color="#ff1744")
                    fig_pos.add_hline(y=trail_stop_actual, line_dash="dot",
                                      line_color="#ff6e40",
                                      annotation_text=f"Trail ${trail_stop_actual:.2f}",
                                      annotation_font_color="#ff6e40")
                    fig_pos.update_layout(
                        paper_bgcolor="#0e1117", plot_bgcolor="#1c2030",
                        font_color="white", height=300,
                        margin=dict(l=0,r=0,t=20,b=0))
                    st.plotly_chart(fig_pos, use_container_width=True)

                # Remove button
                if st.button(f"🗑️ Remove {ticker}", key=f"remove_{idx_p}"):
                    to_remove.append(idx_p)

                st.markdown("")

        # Remove closed positions
        for idx_r in sorted(to_remove, reverse=True):
            st.session_state.positions.pop(idx_r)
        if to_remove:
            st.rerun()

        # ── Portfolio summary ──
        if len(st.session_state.positions) > 1:
            st.markdown("---")
            st.markdown("#### Portfolio summary")
            total_pnl   = sum(
                (get_col(raw_pos,p["ticker"],"Close").iloc[-1] - p["entry"]) * p["shares"]
                for p in st.session_state.positions
                if len(get_col(raw_pos,p["ticker"],"Close")) > 0)
            total_value = sum(
                get_col(raw_pos,p["ticker"],"Close").iloc[-1] * p["shares"]
                for p in st.session_state.positions
                if len(get_col(raw_pos,p["ticker"],"Close")) > 0)
            ps1,ps2,ps3 = st.columns(3)
            ps1.metric("Open positions",  len(st.session_state.positions))
            ps2.metric("Total P&L",       f"${total_pnl:+,.0f}",
                       delta_color="normal" if total_pnl>=0 else "inverse")
            ps3.metric("Total value",     f"${total_value:,.0f}")

st.markdown("---")
st.caption("⚠️ For informational purposes only. Not financial advice.")
