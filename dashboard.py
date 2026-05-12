"""
Equity Quant Dashboard — Simplified High-Sharpe Model
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

# ── FIXED INTERNAL PARAMETERS ─────────────────────────────────────────────────

KELLY_FRAC  = 0.30
MAX_POS     = 0.12
MC_N        = 1000
REGIME_VIX  = 25
MIN_Z       = 0.3
MAX_Z       = 3.0

WEIGHTS = {
    "price_vol":0.30,"sector_rs":0.25,"breadth":0.20,
    "volatility":0.15,"yield_curve":0.10
}

# ── UNIVERSES ─────────────────────────────────────────────────────────────────

FTSE250 = [
    "3IN.L","ABF.L","AEW.L","AGR.L","AIEA.L","AJB.L","ANIC.L","ANP.L","APAX.L","APEO.L",
    "ATG.L","ATM.L","ATST.L","AV.L","AVG.L","AWE.L","AXS.L","BBY.L","BCPT.L","BEG.L",
    "BKG.L","BLTG.L","BME.L","BMS.L","BNKR.L","BOO.L","BOWL.L","BRW.L","BYG.L","CAL.L",
    "CBOX.L","CDFF.L","CEY.L","CHG.L","CLDN.L","CLIG.L","CLW.L","CMCX.L","CNE.L","COPL.L",
    "CTEC.L","CVS.L","CWK.L","DARK.L","DCG.L","DEVO.L","DKG.L","DLG.L","DNO.L","DOCS.L",
    "DPLM.L","DRX.L","DSG.L","DUTY.L","DWF.L","ECOR.L","EDEN.L","ELAS.L","EMG.L","ERM.L",
    "ESNT.L","ESYS.L","ETF.L","EVRB.L","EWI.L","EXL.L","EYES.L","EZJ.L","FAN.L","FCIT.L",
    "FGP.L","FHPE.L","FIGS.L","FIPP.L","FJV.L","FLO.L","FLT.L","FNX.L","FRAS.L","FSFL.L",
    "GAW.L","GBG.L","GDWN.L","GEM.L","GEN.L","GENL.L","GKN.L","GLE.L","GLIF.L","GMG.L",
    "GNS.L","GPOR.L","GRIO.L","GRI.L","GRG.L","GSS.L","GYM.L","HANA.L","HAS.L","HICL.L",
    "HIG.L","HIGG.L","HIK.L","HNE.L","HOME.L","HOTB.L","HRI.L","HRTG.L","HSL.L","HTG.L",
    "HWDN.L","HZD.L","IDS.L","IGR.L","IHR.L","IMI.L","INPP.L","INVP.L","IP.L","IQG.L",
    "ITX.L","JET2.L","JFJ.L","JII.L","JKX.L","JLG.L","JMG.L","JPE.L","JRS.L","JZCP.L",
    "KBC.L","KDNC.L","KGF.L","KNOS.L","KWS.L","LAM.L","LAND.L","LBG.L","LCUS.L","LDG.L",
    "LGB.L","LLPD.L","LMP.L","LPA.L","LRD.L","LSL.L","LTI.L","LXI.L","MAB.L","MARS.L",
    "MBH.L","MCKS.L","MCR.L","MEGA.L","MERC.L","MFXS.L","MHN.L","MILS.L","MKS.L","MLPE.L",
    "MMB.L","MNKS.L","MNTN.L","MOB.L","MRC.L","MRL.L","MRO.L","MSI.L","MTC.L","MTO.L",
    "MTW.L","MXCT.L","MXLS.L","NAH.L","NBB.L","NCC.L","NCYF.L","NMC.L","NTG.L","NVGT.L",
    "OCI.L","OCN.L","OML.L","ONT.L","OPG.L","ORCP.L","OXIG.L","PAGE.L","PCA.L","PCTN.L",
    "PDG.L","PEEL.L","PEG.L","PEN.L","PFG.L","PFDO.L","PGIT.L","PHI.L","PIC.L","PINC.L",
    "PLND.L","PLG.L","PMP.L","PNVD.L","POG.L","PPH.L","PRI.L","PRSR.L","PSH.L","PTG.L",
    "PTEC.L","PXC.L","QQ.L","RACE.L","RCP.L","RDW.L","RGU.L","RKT.L","RMG.L","ROR.L",
    "RPS.L","RQI.L","RSTR.L","SAFE.L","SBRY.L","SDY.L","SEE.L","SFR.L","SGRO.L","SHB.L",
    "SHP.L","SINCE.L","SIR.L","SKG.L","SLA.L","SLPE.L","SMIF.L","SND.L","SNWS.L","SOM.L",
    "SPH.L","SPI.L","SPIT.L","SQZ.L","SRSP.L","SSE.L","STAF.L","STOB.L","STS.L","SUP.L",
    "SUPR.L","SVE.L","SXS.L","SYNT.L","TATE.L","TCAP.L","TCG.L","TET.L","TIG.L","TLW.L",
    "TMPL.L","TRI.L","TRN.L","TRS.L","TUI.L","TWD.L","TYMN.L","UCG.L","UEM.L","ULE.L",
    "UNOG.L","UTG.L","UTV.L","VCT.L","VNET.L","VNT.L","VSVS.L","VTU.L","WAND.L","WEB.L",
    "WEIR.L","WHR.L","WINV.L","WIX.L","WKP.L","WMH.L","WPP.L","WTB.L","XAR.L","XPS.L",
    "YCA.L","ZTF.L",
]

# FTSE 350 = FTSE 100 + FTSE 250
FTSE350 = list(dict.fromkeys(FTSE100 + FTSE250))

# Add FTSE 250 sector assignments (approximate)
FTSE250_SECTOR_MAP = {
    "JET2.L":"XLY","EZJ.L":"XLY","TUI.L":"XLY","BOO.L":"XLY","FRAS.L":"XLY",
    "GYM.L":"XLY","GAW.L":"XLY","KGF.L":"XLY","MKS.L":"XLY","SBRY.L":"XLP",
    "IMI.L":"XLI","WEIR.L":"XLI","GKN.L":"XLI","ROR.L":"XLI","MTO.L":"XLI",
    "TRN.L":"XLI","CHG.L":"XLV","CVS.L":"XLV","HIK.L":"XLV","NMC.L":"XLV",
    "CMCX.L":"XLC","IDS.L":"XLC","WMH.L":"XLC","PAGE.L":"XLC",
    "DLG.L":"XLF","LBG.L":"XLF","AJB.L":"XLF","PFG.L":"XLF","IG.L":"XLF",
    "INVP.L":"XLF","TCAP.L":"XLF","SLA.L":"XLF","JMG.L":"XLF",
    "POG.L":"XLB","CEY.L":"XLB","PEN.L":"XLB","CNE.L":"XLE","TLW.L":"XLE",
    "GBG.L":"XLK","DARK.L":"XLK","CTEC.L":"XLK","ESNT.L":"XLK","NCC.L":"XLK",
    "SGRO.L":"XLRE","LXI.L":"XLRE","SUPR.L":"XLRE","LAND.L":"XLRE",
    "HICL.L":"XLU","INPP.L":"XLU","TRIG.L":"XLU",
    "GRG.L":"XLP","ABF.L":"XLP","TATE.L":"XLP",
}

# Merge FTSE sector maps
FTSE_SECTOR_MAP.update(FTSE250_SECTOR_MAP)

FTSE100 = [
    "SHEL.L","AZN.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L","BATS.L",
    "DGE.L","REL.L","NG.L","VOD.L","LLOY.L","BARC.L","NWG.L","IMB.L",
    "PRU.L","LSEG.L","ABF.L","CRH.L","EXPN.L","FERG.L","HIK.L","HLMA.L",
    "IHG.L","III.L","JD.L","KGF.L","LAND.L","MKS.L","MNG.L","MNDI.L",
    "OCDO.L","PSN.L","RKT.L","RMV.L","SGE.L","SMDS.L","SMIN.L","SSE.L",
    "SVT.L","TSCO.L","TW.L","AAL.L","ADM.L","AGK.L","ANTO.L","AUTO.L",
    "AV.L","AVV.L","BA.L","BEZ.L","BKG.L","BNZL.L","BRBY.L","BT-A.L",
    "CCH.L","CCL.L","CNA.L","CPG.L","CRDA.L","DLN.L","DPLM.L","EDV.L",
    "FRES.L","GLEN.L","HLN.L","HMSO.L","IAG.L","ITRK.L","KIE.L","LGEN.L",
    "MRO.L","NXT.L","PHNX.L","SBRY.L","SGRO.L","SJP.L","SLA.L","SN.L",
    "STAN.L","STJ.L","TATE.L","TUI.L","UU.L","WPP.L","WTB.L",
]

FTSE_SECTOR_MAP = {
    "SHEL.L":"XLE","BP.L":"XLE","MRO.L":"XLE",
    "RIO.L":"XLB","GLEN.L":"XLB","ANTO.L":"XLB","FRES.L":"XLB","AAL.L":"XLB",
    "HSBA.L":"XLF","BARC.L":"XLF","LLOY.L":"XLF","NWG.L":"XLF","STAN.L":"XLF",
    "LSEG.L":"XLF","SJP.L":"XLF","LGEN.L":"XLF","PRU.L":"XLF","AV.L":"XLF",
    "MNG.L":"XLF","III.L":"XLF","PHNX.L":"XLF","SLA.L":"XLF",
    "AZN.L":"XLV","GSK.L":"XLV","HIK.L":"XLV","HLN.L":"XLV","STJ.L":"XLV",
    "EXPN.L":"XLK","SGE.L":"XLK","AVV.L":"XLK","REL.L":"XLK","DPLM.L":"XLK",
    "HLMA.L":"XLK","SMIN.L":"XLK","ITRK.L":"XLK",
    "ULVR.L":"XLP","BATS.L":"XLP","IMB.L":"XLP","DGE.L":"XLP","ABF.L":"XLP",
    "TSCO.L":"XLP","SBRY.L":"XLP","MKS.L":"XLP","TATE.L":"XLP","CCH.L":"XLP",
    "IHG.L":"XLY","JD.L":"XLY","NXT.L":"XLY","BRBY.L":"XLY","RMV.L":"XLY",
    "AUTO.L":"XLY","OCDO.L":"XLY","TUI.L":"XLY","IAG.L":"XLY","CCL.L":"XLY",
    "CRH.L":"XLI","FERG.L":"XLI","BA.L":"XLI","RKT.L":"XLI","MNDI.L":"XLI",
    "SMDS.L":"XLI","PSN.L":"XLI","KIE.L":"XLI","BEZ.L":"XLI","WTB.L":"XLI",
    "NG.L":"XLU","SSE.L":"XLU","SVT.L":"XLU","UU.L":"XLU",
    "VOD.L":"XLC","BT-A.L":"XLC","WPP.L":"XLC",
    "LAND.L":"XLRE","SGRO.L":"XLRE","HMSO.L":"XLRE",
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
    "WFC","HON","BMY","CVS","SBUX","F","GM","UBER","ABNB","COIN",
    "SHOP","SPOT","RBLX","DDOG","ZS","CRWD","PANW","OKTA","FTNT","CHKP",
    "RCL","CCL","MAR","HLT","DAL","UAL","AAL","LUV","COP","EOG",
    "SLB","HAL","OXY","DVN","MPC","PSX","VLO","PNC","TFC","COF",
    "DFS","SYF","ALLY","CFG","FITB","KEY","BK","STT","TROW","FIS",
    "FISV","GPN","PYPL","HCA","THC","MRNA","BNTX","PFE","NVO","ZBH",
    "INTC","MU","AMAT","LRCX","KLAC","CDNS","SNPS","WDAY","VEEV","HUBS",
    "DIS","CMCSA","CHTR","TTWO","EA","EBAY","ETSY","YUM","QSR","DRI",
    "GIS","MMM","PH","ROK","LMT","NOC","GD","LHX","UPS","XPO",
    "JBHT","ODFL","EXC","AEP","XEL","WEC","DTE","AWK","AMT","CCI",
    "EQIX","DLR","PSA","NEM","FCX","SCCO","PPG",
]

SECTOR_MAP = {
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","GOOGL":"XLC","GOOG":"XLC",
    "META":"XLC","AMZN":"XLY","TSLA":"XLY","JPM":"XLF","UNH":"XLV",
    "XOM":"XLE","V":"XLF","AVGO":"XLK","PG":"XLP","MA":"XLF","JNJ":"XLV",
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
    "PLD":"XLRE","NKE":"XLY","WFC":"XLF","HON":"XLI","BMY":"XLV",
    "CVS":"XLV","SBUX":"XLY","F":"XLY","GM":"XLY","UBER":"XLY",
    "ABNB":"XLY","COIN":"XLF","SHOP":"XLK","SPOT":"XLC","RBLX":"XLC",
    "DDOG":"XLK","ZS":"XLK","CRWD":"XLK","PANW":"XLK","OKTA":"XLK",
    "FTNT":"XLK","CHKP":"XLK","RCL":"XLY","CCL":"XLY","MAR":"XLY",
    "HLT":"XLY","DAL":"XLI","UAL":"XLI","AAL":"XLI","LUV":"XLI",
    "COP":"XLE","EOG":"XLE","SLB":"XLE","HAL":"XLE","OXY":"XLE",
    "DVN":"XLE","MPC":"XLE","PSX":"XLE","VLO":"XLE","PNC":"XLF",
    "TFC":"XLF","COF":"XLF","DFS":"XLF","SYF":"XLF","ALLY":"XLF",
    "CFG":"XLF","FITB":"XLF","KEY":"XLF","BK":"XLF","STT":"XLF",
    "TROW":"XLF","FIS":"XLK","FISV":"XLK","GPN":"XLK","PYPL":"XLK",
    "HCA":"XLV","THC":"XLV","MRNA":"XLV","BNTX":"XLV","PFE":"XLV",
    "NVO":"XLV","ZBH":"XLV","INTC":"XLK","MU":"XLK","AMAT":"XLK",
    "LRCX":"XLK","KLAC":"XLK","CDNS":"XLK","SNPS":"XLK","WDAY":"XLK",
    "VEEV":"XLK","HUBS":"XLK","DIS":"XLC","CMCSA":"XLC","CHTR":"XLC",
    "TTWO":"XLC","EA":"XLC","EBAY":"XLY","ETSY":"XLY","YUM":"XLY",
    "QSR":"XLY","DRI":"XLY","GIS":"XLP","MMM":"XLI","PH":"XLI",
    "ROK":"XLI","LMT":"XLI","NOC":"XLI","GD":"XLI","LHX":"XLI",
    "UPS":"XLI","XPO":"XLI","JBHT":"XLI","ODFL":"XLI","EXC":"XLU",
    "AEP":"XLU","XEL":"XLU","WEC":"XLU","DTE":"XLU","AWK":"XLU",
    "AMT":"XLRE","CCI":"XLRE","EQIX":"XLRE","DLR":"XLRE","PSA":"XLRE",
    "NEM":"XLB","FCX":"XLB","SCCO":"XLB","PPG":"XLB","BRK-B":"XLF",
}

SECTOR_NAMES = {
    "XLK":"Technology","XLC":"Communication","XLY":"Consumer Disc.",
    "XLF":"Financials","XLE":"Energy","XLV":"Healthcare",
    "XLP":"Consumer Staples","XLI":"Industrials",
    "XLU":"Utilities","XLB":"Materials","XLRE":"Real Estate",
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    user_input = st.text_input(
        "Watchlist",
        value="SHEL.L,AZN.L,HSBA.L,BP.L,LLOY.L,BARC.L,GSK.L,ULVR.L,AAPL,MSFT,NVDA,JPM",
        key="sidebar_watchlist"
    )
    WATCHLIST = [t.strip().upper() for t in user_input.split(",") if t.strip()]

    st.markdown("---")
    st.markdown("### Signal quality")
    quality = st.select_slider(
        "Win probability required",
        options=["Moderate (≥55%)", "Strong (≥60%)", "Very Strong (≥70%)"],
        value="Strong (≥60%)"
    )
    MIN_WIN_PROB = {
        "Moderate (≥55%)": 0.55,
        "Strong (≥60%)":   0.60,
        "Very Strong (≥70%)": 0.70
    }[quality]

    st.markdown("---")
    with st.expander("ℹ️ How it works"):
        st.caption(
            "5 statistical layers run internally:\n\n"
            "**L1** Bayesian win probability\n"
            "**L2** Market regime (Bull only)\n"
            "**L3** Z-score entry timing\n"
            "**L4** Kelly position sizing (30%)\n"
            "**L5** Monte Carlo downside check\n\n"
            "One control is exposed because all other "
            "parameters are fixed at Sharpe-optimal values."
        )
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

def get_sector(ticker):
    if ticker.endswith(".L"): return FTSE_SECTOR_MAP.get(ticker, "SPY")
    return SECTOR_MAP.get(ticker, "SPY")

@st.cache_data(ttl=300)
def fetch_data(tickers_key, days=420):
    end = datetime.today(); start = end - timedelta(days=days)
    return yf.download(list(tickers_key), start=start, end=end,
                       auto_adjust=True, progress=False, group_by="ticker")

# ── STATISTICAL ENGINE ────────────────────────────────────────────────────────

def compute_dms(c, v, vix_s, spy_s, sc, t10, t5):
    avg_vol   = v.rolling(20).mean().iloc[-1]
    vol_ratio = v.iloc[-1]/avg_vol if avg_vol > 0 else 1.0
    pv        = normalise((c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*vol_ratio, -0.05, 0.05)
    vix_score = 100 - normalise(vix_s.iloc[-1], 10, 40)
    hi=c.rolling(2).max(); lo=c.rolling(2).min()
    atr_pct   = ((hi-lo)/c).rolling(14).mean().iloc[-1]*100
    vlt       = (vix_score + (100-normalise(atr_pct, 0.5, 5.0))) / 2
    brd       = normalise(spy_s.iloc[-1]/spy_s.rolling(50).mean().iloc[-1], 0.90, 1.10)
    try:    yld = normalise(t10.iloc[-1]-t5.iloc[-1], -0.5, 2.0)
    except: yld = 50.0
    w  = min(20, len(c)-1)
    sr = (c.iloc[-1]/c.iloc[-w])-1; er=(sc.iloc[-1]/sc.iloc[-w])-1
    rs = normalise((1+sr)/((1+er) if (1+er)!=0 else 1), 0.70, 1.30)
    return round(WEIGHTS["price_vol"]*pv + WEIGHTS["sector_rs"]*rs +
                 WEIGHTS["breadth"]*brd  + WEIGHTS["volatility"]*vlt +
                 WEIGHTS["yield_curve"]*yld, 2)

def bayesian_win_prob(c, dms_now, window=20, horizon=10):
    ret   = c.pct_change().dropna()
    rm    = ret.rolling(window).mean(); rs=ret.rolling(window).std()
    proxy = ((rm/rs.replace(0,np.nan)).clip(-3,3)*16.67+50).dropna()
    fwd   = c.pct_change(horizon).shift(-horizon)
    aln   = pd.concat([proxy,fwd],axis=1).dropna(); aln.columns=["d","f"]
    lo=max(0,dms_now-10); hi=min(100,dms_now+10)
    b = aln[(aln["d"]>=lo)&(aln["d"]<=hi)]
    if len(b)<10: return 0.50, len(b)
    return round(float((b["f"]>0).mean()),3), len(b)

def classify_regime(vix_s, spy_s):
    vix=vix_s.iloc[-1]; spy=spy_s.iloc[-1]
    spy200=spy_s.rolling(200).mean().iloc[-1]; above=spy>spy200
    if above and vix<15:          return "Bull quiet",    True,  2.1, vix, spy, spy200
    if above and vix<REGIME_VIX:  return "Bull volatile", True,  0.8, vix, spy, spy200
    if not above and vix<20:      return "Bear quiet",    False,-0.3, vix, spy, spy200
    return "Bear volatile", False, -2.4, vix, spy, spy200

def zscore_entry(c, window=20):
    ret=c.pct_change().dropna()
    mu=ret.rolling(window).mean().iloc[-1]; sd=ret.rolling(window).std().iloc[-1]
    if sd==0 or pd.isna(sd): return 0.0
    return round(float((ret.iloc[-1]-mu)/sd), 2)

def kelly_size(win_prob, avg_win=0.03, avg_loss=0.015):
    if avg_loss==0: return 0.0
    b=avg_win/avg_loss; q=1-win_prob
    return round(min(max((b*win_prob-q)/b*KELLY_FRAC, 0), MAX_POS), 4)

def monte_carlo(c, kelly, capital=100000, horizon=10):
    ret   = c.pct_change().dropna().values
    entry = c.iloc[-1]; shares=(capital*kelly)/entry
    oc    = np.array([
        (entry*np.prod(1+np.random.choice(ret,size=horizon,replace=True))-entry)*shares
        for _ in range(MC_N)])
    p25_floor = -abs(np.percentile(oc,50))
    return (round(float(np.percentile(oc,25)),0),
            round(float(np.percentile(oc,50)),0),
            round(float(np.percentile(oc,75)),0),
            round(float((oc<0).mean()),3), oc, p25_floor)

def evaluate(ticker, c, v, vix_s, spy_s, sc, t10, t5):
    dms              = compute_dms(c,v,vix_s,spy_s,sc,t10,t5)
    win_prob, n_samp = bayesian_win_prob(c, dms)
    regime,trd,exp_ret,vix_now,spy_now,spy_200 = classify_regime(vix_s,spy_s)
    z                = zscore_entry(c)
    kelly            = kelly_size(win_prob)
    p25,p50,p75,ploss,mc_oc,p25f = (monte_carlo(c,kelly) if kelly>0
                                     else (0,0,0,1.0,np.array([0]),0))
    l1=win_prob>=MIN_WIN_PROB; l2=trd; l3=MIN_Z<=z<=MAX_Z
    l4=kelly>0.01;             l5=p25>p25f
    all_pass = l1 and l2 and l3 and l4 and l5
    atr=(c.rolling(2).max()-c.rolling(2).min()).rolling(14).mean().iloc[-1]
    return {
        "Ticker":ticker,"Price":round(float(c.iloc[-1]),2),
        "DMS":dms,"Sector":SECTOR_NAMES.get(get_sector(ticker),"—"),
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

# ── LOAD WATCHLIST ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def run_watchlist(wl_key, win_prob_key):
    all_tix=tuple(set(list(wl_key)+list(SECTOR_MAP.values())+
                      list(FTSE_SECTOR_MAP.values())+["SPY","^VIX","^TNX","^FVX"]))
    raw   = fetch_data(all_tix)
    vix_s = get_col(raw,"^VIX","Close"); spy_s=get_col(raw,"SPY","Close")
    t10   = get_col(raw,"^TNX","Close"); t5   =get_col(raw,"^FVX","Close")
    out   = []
    for ticker in wl_key:
        c=get_col(raw,ticker,"Close"); v=get_col(raw,ticker,"Volume")
        sc=get_col(raw,get_sector(ticker),"Close")
        if len(c)<60: continue
        try: out.append(evaluate(ticker,c,v,vix_s,spy_s,sc,t10,t5))
        except: continue
    return out

with st.spinner("Analysing..."):
    results = run_watchlist(tuple(WATCHLIST), MIN_WIN_PROB)

if not results:
    st.error("No data returned. Check watchlist."); st.stop()

trades = [r for r in results if r["all_pass"]]
r0     = results[0]

# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-bar">
    <h2 style="color:white;margin:0">📊 Equity Quant Model</h2>
    <p style="color:#90caf9;margin:4px 0 0">
    High-Sharpe signal engine — only trade when the odds are provably in your favour</p>
</div>""", unsafe_allow_html=True)

rc = "#00e676" if r0["l2"] else "#ff1744"
spy_diff = ((r0["spy_now"]/r0["spy_200"])-1)*100
st.markdown(
    f'<div style="background:#1c2030;border-left:4px solid {rc};'
    f'padding:10px 16px;border-radius:6px;margin-bottom:16px;">'
    f'<span style="color:{rc};font-weight:700;font-size:15px">'
    f'{"✅" if r0["l2"] else "❌"} Market regime: {r0["regime"]}</span>'
    f'<span style="color:#888;font-size:13px;margin-left:16px">'
    f'VIX {r0["vix_now"]:.1f} &nbsp;|&nbsp; '
    f'SPY {spy_diff:+.1f}% vs 200MA &nbsp;|&nbsp; '
    f'Expected return: {r0["exp_ret"]:+.1f}%/month &nbsp;|&nbsp; '
    f'Signal quality: <b style="color:white">{quality}</b></span></div>',
    unsafe_allow_html=True)

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Stocks analysed",     len(results))
k2.metric("Trade signals today",  len(trades),
          delta="Stand aside" if not trades else "Act on these")
k3.metric("Avg win probability",  f"{np.mean([r['win_prob'] for r in results])*100:.1f}%")
k4.metric("Avg expected value",   f"${np.mean([r['ev'] for r in results]):,.0f}")
k5.metric("Kelly position size",  f"{KELLY_FRAC*100:.0f}% fraction",
          delta="Sharpe-optimised")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────

tab0,tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🔭 Screener","🎯 Decisions","📈 Charts",
    "🎲 Monte Carlo","🔁 Backtest","📂 Positions"])

# ── TAB 0: SCREENER ───────────────────────────────────────────────────────────

with tab0:
    st.subheader("🔭 Universe Screener")
    st.caption("Scan FTSE 100 or S&P 500 through pre-filters then the 5-layer model.")

    sc1,sc2,sc3 = st.columns(3)
    with sc1:
        universe_choice = st.selectbox("Universe", [
            "FTSE 350 (100+250)","FTSE 100","FTSE 250",
            "S&P 500",
            "FTSE — Financials","FTSE — Healthcare","FTSE — Energy",
            "FTSE — Technology","FTSE — Consumer",
            "S&P — Technology (XLK)","S&P — Financials (XLF)",
            "S&P — Healthcare (XLV)","S&P — Energy (XLE)",
            "S&P — Industrials (XLI)","S&P — Consumer Disc. (XLY)",
            "Custom list"
        ], key="screener_universe")
    with sc2:
        min_vol_m    = st.slider("Min avg volume (M/day)", 0.1, 5.0, 0.5, 0.1,
                                 key="screener_vol")
    with sc3:
        screen_top_n = st.slider("Show top N results", 3, 20, 10,
                                 key="screener_topn")

    if universe_choice == "Custom list":
        custom_input = st.text_input("Tickers (comma-separated)",
                                     value="SHEL.L,AZN.L,HSBA.L,AAPL,MSFT,NVDA",
                                     key="screener_custom")
        screen_univ  = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    elif universe_choice == "FTSE 350 (100+250)":
        screen_univ  = FTSE350
    elif universe_choice == "FTSE 100":
        screen_univ  = FTSE100
    elif universe_choice == "FTSE 250":
        screen_univ  = FTSE250
    elif universe_choice == "S&P 500":
        screen_univ  = list(dict.fromkeys(SP500))
    elif universe_choice.startswith("FTSE —"):
        label   = universe_choice.replace("FTSE — ","").strip()
        etf_map = {"Financials":"XLF","Healthcare":"XLV","Energy":"XLE",
                   "Technology":"XLK","Consumer":"XLP"}
        etf         = etf_map.get(label,"XLF")
        screen_univ = [t for t,s in FTSE_SECTOR_MAP.items() if s==etf]
    else:
        parts       = universe_choice.split("(")
        etf         = parts[1].replace(")","").strip() if len(parts)>1 else "XLK"
        screen_univ = [t for t,s in SECTOR_MAP.items() if s==etf]

    is_ftse     = any(t.endswith(".L") for t in screen_univ[:5])
    screen_univ = list(dict.fromkeys(screen_univ))

    st.info(f"{'🇬🇧 FTSE' if is_ftse else '🇺🇸 S&P'} — "
            f"**{len(screen_univ)} stocks** → pre-filter → 5-layer model → "
            f"top {screen_top_n}  |  Signal quality: **{quality}**")

    if st.button("🔭 Run Screener", type="primary", use_container_width=True,
                 key="run_screener_btn"):
        prog = st.progress(0, text="Fetching data...")

        @st.cache_data(ttl=600)
        def fetch_screen(tickers_key):
            all_tix=list(set(list(tickers_key)+
                             list(set(FTSE_SECTOR_MAP.values()))+
                             list(set(SECTOR_MAP.values()))+
                             ["SPY","^VIX","^TNX","^FVX"]))
            end=datetime.today(); start=end-timedelta(days=420)
            return yf.download(all_tix,start=start,end=end,
                               auto_adjust=True,progress=False,group_by="ticker")

        raw_sc = fetch_screen(tuple(screen_univ))
        vix_sc = get_col(raw_sc,"^VIX","Close"); spy_sc=get_col(raw_sc,"SPY","Close")
        t10_sc = get_col(raw_sc,"^TNX","Close"); t5_sc =get_col(raw_sc,"^FVX","Close")

        prog.progress(20, text="Pre-filtering...")
        regime_sc, tradeable_sc, *_ = classify_regime(vix_sc, spy_sc)

        pf_passed=[]
        for ticker in screen_univ:
            c=get_col(raw_sc,ticker,"Close"); v=get_col(raw_sc,ticker,"Volume")
            if len(c)<55 or len(v)<20: continue
            if c.iloc[-1]<1:           continue
            if v.rolling(20).mean().iloc[-1]<min_vol_m*1e6: continue
            if c.iloc[-1]<c.rolling(50).mean().iloc[-1]:    continue
            if (c.iloc[-1]/c.iloc[-20])-1<=0:               continue
            pf_passed.append(ticker)

        prog.progress(40, text=f"{len(pf_passed)} passed pre-filter...")

        sc_results=[]
        for i,ticker in enumerate(pf_passed):
            try:
                c=get_col(raw_sc,ticker,"Close"); v=get_col(raw_sc,ticker,"Volume")
                sc=get_col(raw_sc,get_sector(ticker),"Close")
                if len(c)<60: continue
                dms    = compute_dms(c,v,vix_sc,spy_sc,sc,t10_sc,t5_sc)
                wp,n_s = bayesian_win_prob(c,dms)
                z      = zscore_entry(c)
                kel    = kelly_size(wp)
                p25,p50,p75,ploss,_,p25f = (monte_carlo(c,kel) if kel>0
                                             else (0,0,0,1.0,None,0))
                _rg,trd,*_ = classify_regime(vix_sc,spy_sc)
                l1=wp>=MIN_WIN_PROB; l2=trd
                l3=MIN_Z<=z<=MAX_Z;  l4=kel>0.01; l5=p25>p25f
                atr=(c.rolling(2).max()-c.rolling(2).min()).rolling(14).mean().iloc[-1]
                sc_results.append({
                    "Ticker":ticker,"Price":round(float(c.iloc[-1]),2),
                    "Win Prob":f"{wp*100:.1f}%","Z-Score":round(z,2),
                    "Kelly %":f"{kel*100:.1f}%",
                    "EV $":p50,"P25 $":p25,"P75 $":p75,
                    "P(loss)":f"{ploss*100:.0f}%",
                    "All pass":l1 and l2 and l3 and l4 and l5,
                    "L1":("✅" if l1 else "❌"),"L2":("✅" if l2 else "❌"),
                    "L3":("✅" if l3 else "❌"),"L4":("✅" if l4 else "❌"),
                    "L5":("✅" if l5 else "❌"),
                    "Stop":round(float(c.iloc[-1])-1.5*float(atr),2),
                    "Sector":SECTOR_NAMES.get(get_sector(ticker),"—"),
                })
            except: pass
            prog.progress(40+int(55*(i+1)/max(len(pf_passed),1)),
                          text=f"Scoring {ticker}...")

        prog.progress(100, text="Done.")

        if not sc_results:
            st.warning("No results generated.")
        else:
            sc_df   = pd.DataFrame(sc_results).sort_values("EV $",ascending=False)
            winners = sc_df[sc_df["All pass"]]
            near    = sc_df[~sc_df["All pass"]].head(10)

            s1,s2,s3,s4 = st.columns(4)
            s1.metric("Scanned",       len(screen_univ))
            s2.metric("Pre-filtered",  len(pf_passed))
            s3.metric("Trade signals", len(winners))
            s4.metric("Regime",        regime_sc,
                      delta="✅ Tradeable" if tradeable_sc else "❌ Bear",
                      delta_color="normal" if tradeable_sc else "inverse")

            st.markdown("---")

            if len(winners):
                st.success(f"✅ **Trade candidates:** "
                           f"{', '.join(winners['Ticker'].tolist())}")

                def ch(v):
                    return ("color:#00e676;font-weight:bold" if v=="✅"
                            else "color:#ff1744;font-weight:bold")
                def ce(v):
                    return f"color:{'#00e676' if v>0 else '#ff1744'};font-weight:bold"

                dcols=["Ticker","Price","Win Prob","Z-Score","Kelly %",
                       "P25 $","EV $","Stop","Sector","L1","L2","L3","L4","L5"]
                st.dataframe(
                    winners[dcols].head(screen_top_n).style
                        .map(ch,subset=["L1","L2","L3","L4","L5"])
                        .map(ce,subset=["EV $"]),
                    use_container_width=True,
                    height=min(80+len(winners)*40,400))

                fig_ev=px.bar(
                    winners.head(screen_top_n).sort_values("EV $"),
                    x="EV $",y="Ticker",orientation="h",
                    color="Win Prob",title="Expected value by stock",
                    height=max(250,len(winners)*50))
                fig_ev.update_layout(paper_bgcolor="#0e1117",
                                     plot_bgcolor="#1c2030",font_color="white")
                st.plotly_chart(fig_ev,use_container_width=True)
            else:
                st.warning("❌ No stocks pass all 5 layers today. "
                           "Try Moderate quality or check back tomorrow.")

            if len(near):
                st.markdown("#### Near misses")
                st.dataframe(
                    near[["Ticker","Price","Win Prob","Z-Score",
                          "EV $","L1","L2","L3","L4","L5","Sector"]],
                    use_container_width=True,height=280)

            st.download_button("⬇️ Download results",sc_df.to_csv(index=False),
                               f"screener_{datetime.today().strftime('%Y%m%d')}.csv",
                               "text/csv",key="screener_download")

# ── TAB 1: DECISIONS ──────────────────────────────────────────────────────────

with tab1:
    if trades:
        st.success(f"✅ **Trade today ({len(trades)}):** "
                   f"{', '.join([r['Ticker'] for r in trades])}")
    else:
        st.warning("❌ No stocks pass all 5 layers today. Stand aside.")

    st.markdown("#### Full breakdown — your watchlist")

    for r in sorted(results, key=lambda x: -x["ev"]):
        border="#00e676" if r["all_pass"] else "#333"
        c0,c1,c2,c3,c4,c5,c6 = st.columns([1.2,1.4,1.4,1,1.4,1.2,1])
        with c0:
            st.markdown(
                f'<div style="background:#1c2030;border-left:3px solid {border};'
                f'padding:8px 12px;border-radius:6px;">'
                f'<div style="color:white;font-weight:700;font-size:16px">{r["Ticker"]}</div>'
                f'<div style="color:#888;font-size:11px">${r["Price"]:.2f} · {r["Sector"]}</div>'
                f'</div>', unsafe_allow_html=True)
        with c1:
            col="#00e676" if r["l1"] else "#ff1744"
            st.markdown(
                f'<div style="background:#1c2030;border-radius:6px;padding:8px 12px;">'
                f'<div style="color:#90caf9;font-size:10px">Win probability</div>'
                f'<div style="color:{col};font-size:18px;font-weight:700">'
                f'{r["win_prob"]*100:.1f}% {"✅" if r["l1"] else "❌"}</div>'
                f'<div style="color:#555;font-size:10px">'
                f'Need ≥{MIN_WIN_PROB*100:.0f}% · n={r["n_samp"]}</div></div>',
                unsafe_allow_html=True)
        with c2:
            col="#00e676" if r["l2"] else "#ff1744"
            st.markdown(
                f'<div style="background:#1c2030;border-radius:6px;padding:8px 12px;">'
                f'<div style="color:#90caf9;font-size:10px">Market regime</div>'
                f'<div style="color:{col};font-size:14px;font-weight:700">'
                f'{r["regime"]} {"✅" if r["l2"] else "❌"}</div>'
                f'<div style="color:#555;font-size:10px">'
                f'{r["exp_ret"]:+.1f}%/month expected</div></div>',
                unsafe_allow_html=True)
        with c3:
            col="#00e676" if r["l3"] else "#ff1744"
            st.markdown(
                f'<div style="background:#1c2030;border-radius:6px;padding:8px 12px;">'
                f'<div style="color:#90caf9;font-size:10px">Z-score</div>'
                f'<div style="color:{col};font-size:18px;font-weight:700">'
                f'{r["z"]:.2f} {"✅" if r["l3"] else "❌"}</div>'
                f'<div style="color:#555;font-size:10px">Entry timing</div></div>',
                unsafe_allow_html=True)
        with c4:
            col="#00e676" if r["l4"] else "#ff1744"
            st.markdown(
                f'<div style="background:#1c2030;border-radius:6px;padding:8px 12px;">'
                f'<div style="color:#90caf9;font-size:10px">Position size</div>'
                f'<div style="color:{col};font-size:18px;font-weight:700">'
                f'{r["kelly"]*100:.1f}% {"✅" if r["l4"] else "❌"}</div>'
                f'<div style="color:#555;font-size:10px">30% Kelly</div></div>',
                unsafe_allow_html=True)
        with c5:
            col="#00e676" if r["l5"] else "#ff1744"
            st.markdown(
                f'<div style="background:#1c2030;border-radius:6px;padding:8px 12px;">'
                f'<div style="color:#90caf9;font-size:10px">Downside check</div>'
                f'<div style="color:{col};font-size:16px;font-weight:700">'
                f'${r["p25"]:,.0f} {"✅" if r["l5"] else "❌"}</div>'
                f'<div style="color:#555;font-size:10px">'
                f'P(loss) {r["ploss"]*100:.0f}%</div></div>',
                unsafe_allow_html=True)
        with c6:
            bg="#00e676" if r["all_pass"] else "#1c2030"
            txt="#003300" if r["all_pass"] else "#555"
            st.markdown(
                f'<div style="background:{bg};border-radius:6px;'
                f'padding:8px 12px;text-align:center;">'
                f'<div style="color:{txt};font-size:13px;font-weight:700">'
                f'{"TRADE" if r["all_pass"] else "SKIP"}</div>'
                f'<div style="color:{txt};font-size:11px">'
                f'EV ${r["ev"]:+,.0f}</div></div>',
                unsafe_allow_html=True)
        st.markdown("")

# ── TAB 2: CHARTS ─────────────────────────────────────────────────────────────

with tab2:
    sel  = st.selectbox("Select stock",[r["Ticker"] for r in results],key="chart_sel")
    r2   = next(x for x in results if x["Ticker"]==sel)
    cls  = r2["close_s"]

    fig_p=go.Figure()
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
        title=(f"{sel}  |  Win prob {r2['win_prob']*100:.1f}%  |  "
               f"Kelly {r2['kelly']*100:.1f}%  |  "
               f"{'✅ TRADE' if r2['all_pass'] else '❌ SKIP'}"),
        paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",font_color="white",height=460)
    st.plotly_chart(fig_p,use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        fig_g=go.Figure(go.Indicator(
            mode="gauge+number",value=r2["win_prob"]*100,number={"suffix":"%"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":"#00e676" if r2["l1"] else "#ff1744"},
                "steps":[{"range":[0,45],"color":"#b71c1c"},
                          {"range":[45,55],"color":"#f57f17"},
                          {"range":[55,70],"color":"#2e7d32"},
                          {"range":[70,100],"color":"#1b5e20"}],
                "threshold":{"line":{"color":"white","width":3},
                             "value":MIN_WIN_PROB*100}},
            title={"text":f"Win probability (need ≥{MIN_WIN_PROB*100:.0f}%)",
                   "font":{"color":"white"}}))
        fig_g.update_layout(paper_bgcolor="#0e1117",font_color="white",height=260)
        st.plotly_chart(fig_g,use_container_width=True)

    with col2:
        ret_s  = cls.pct_change().dropna()
        rm     = ret_s.rolling(20).mean(); rs2=ret_s.rolling(20).std()
        proxy  = ((rm/rs2.replace(0,np.nan)).clip(-3,3)*16.67+50).dropna()
        fwd    = cls.pct_change(10).shift(-10)
        aln    = pd.concat([proxy,fwd],axis=1).dropna(); aln.columns=["d","f"]
        rwp    = aln["f"].gt(0).rolling(30).mean()
        fig_wp = go.Figure()
        fig_wp.add_trace(go.Scatter(x=rwp.index,y=rwp*100,fill="tozeroy",
                                    fillcolor="rgba(33,150,243,0.15)",
                                    line=dict(color="#2196F3",width=1.5)))
        fig_wp.add_hline(y=MIN_WIN_PROB*100,line_dash="dash",line_color="#00e676",
                         annotation_text=f"Threshold {MIN_WIN_PROB*100:.0f}%")
        fig_wp.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                             font_color="white",height=260,
                             yaxis=dict(range=[0,100],title="Win prob %"),
                             title="Rolling win probability over time")
        st.plotly_chart(fig_wp,use_container_width=True)

    st.markdown("#### Monte Carlo — 1,000 simulated outcomes")
    mc_oc=r2["mc_outcomes"]
    fig_mc=go.Figure()
    fig_mc.add_trace(go.Histogram(x=mc_oc,nbinsx=60,
                                  marker_color="#2196F3",opacity=0.75))
    for pct,col,lbl in [(25,"#ff6e40","P25"),(50,"#ffd740","P50"),(75,"#00e676","P75")]:
        v2=np.percentile(mc_oc,pct)
        fig_mc.add_vline(x=v2,line_dash="dash",line_color=col,
                         annotation_text=f"{lbl}: ${v2:,.0f}",annotation_font_color=col)
    fig_mc.add_vline(x=0,line_color="#555",line_width=1)
    fig_mc.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                         font_color="white",height=280,
                         xaxis_title="10-day P&L ($)",yaxis_title="Frequency",
                         title=dict(text=f"{sel} — P(loss): {r2['ploss']*100:.1f}%",
                                    font=dict(color="white")))
    st.plotly_chart(fig_mc,use_container_width=True)

    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.metric("P25 (bad case)",  f"${r2['p25']:,.0f}")
    mc2.metric("P50 (median)",    f"${r2['p50']:,.0f}")
    mc3.metric("P75 (good case)", f"${r2['p75']:,.0f}")
    mc4.metric("P(loss)",         f"{r2['ploss']*100:.1f}%")

# ── TAB 3: MONTE CARLO EXPLORER ───────────────────────────────────────────────

with tab3:
    st.subheader("🎲 Monte Carlo Explorer")
    st.caption("Simulate trade outcomes at different position sizes and holding periods.")

    mc_ticker = st.selectbox("Stock",[r["Ticker"] for r in results],key="mc_sel")
    mc_r      = next(x for x in results if x["Ticker"]==mc_ticker)
    mc_cls    = mc_r["close_s"]

    mc1c,mc2c,mc3c = st.columns(3)
    with mc1c: mc_capital = st.number_input("Portfolio ($)",value=100000,
                                             step=10000,key="mc_cap")
    with mc2c: mc_pos_pct = st.slider("Position %",1,25,
                                       max(1,int(mc_r["kelly"]*100)),key="mc_pos")
    with mc3c: mc_horizon = st.slider("Hold days",5,60,15,key="mc_hor")

    if st.button("▶️ Run simulation",type="primary",key="mc_run"):
        ret_mc=mc_cls.pct_change().dropna().values
        entry=mc_cls.iloc[-1]; shares=(mc_capital*(mc_pos_pct/100))/entry
        sims=np.array([
            (entry*np.prod(1+np.random.choice(ret_mc,size=mc_horizon,replace=True))-entry)*shares
            for _ in range(MC_N)])
        s1,s2,s3,s4=st.columns(4)
        s1.metric("P10",f"${np.percentile(sims,10):,.0f}")
        s2.metric("P25",f"${np.percentile(sims,25):,.0f}")
        s3.metric("P50",f"${np.percentile(sims,50):,.0f}")
        s4.metric("P75",f"${np.percentile(sims,75):,.0f}")
        fig_sim=go.Figure()
        fig_sim.add_trace(go.Histogram(x=sims,nbinsx=80,
                                       marker_color="#2196F3",opacity=0.75))
        for pct,col,lbl in [(10,"#ff1744","P10"),(25,"#ff6e40","P25"),
                             (50,"#ffd740","P50"),(75,"#00e676","P75")]:
            v3=np.percentile(sims,pct)
            fig_sim.add_vline(x=v3,line_dash="dash",line_color=col,
                              annotation_text=f"{lbl}: ${v3:,.0f}",annotation_font_color=col)
        fig_sim.add_vline(x=0,line_color="#555",line_width=1.5)
        fig_sim.update_layout(
            paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",font_color="white",height=360,
            title=dict(text=(f"{mc_ticker} — {mc_pos_pct}% position  |  "
                             f"{mc_horizon}d hold  |  "
                             f"P(loss): {(sims<0).mean()*100:.1f}%"),
                       font=dict(color="white")),
            xaxis_title="P&L ($)",yaxis_title="Frequency")
        st.plotly_chart(fig_sim,use_container_width=True)
        kelly_rec=kelly_size(mc_r["win_prob"])*100
        if mc_pos_pct>kelly_rec*1.5:
            st.warning(f"⚠️ {mc_pos_pct}% exceeds Kelly ({kelly_rec:.1f}%). Sharpe will suffer.")
        else:
            st.success(f"✅ Within Kelly bounds (recommended: {kelly_rec:.1f}%)")

# ── TAB 4: BACKTEST ───────────────────────────────────────────────────────────

with tab4:
    st.subheader("🔁 Backtest")

    bc1,bc2,bc3 = st.columns(3)
    with bc1:
        bt_tick = st.text_input("Ticker",value="SHEL.L",key="bt_ticker_input").upper()
    with bc2:
        bt_period = st.selectbox("Period",
            ["5 years","3 years","2 years","1 year","Custom"],
            index=0,key="bt_period")
    with bc3:
        bt_cap = st.number_input("Capital ($)",value=100000,step=10000,key="bt_cap")

    if bt_period=="Custom":
        cc1,cc2=st.columns(2)
        with cc1: bt_start=st.date_input("Start",
                            value=datetime.today()-timedelta(days=5*365),
                            key="bt_start")
        with cc2: bt_end=st.date_input("End",value=datetime.today(),key="bt_end")
    else:
        years=int(bt_period.split()[0])
        bt_start=datetime.today()-timedelta(days=years*365)
        bt_end  =datetime.today()

    st.info(f"**{bt_tick}** · "
            f"{bt_start.strftime('%d %b %Y') if hasattr(bt_start,'strftime') else bt_start}"
            f" → today · {bt_period} · {quality}")

    # ── Slippage & cost controls ──
    sl1,sl2,sl3 = st.columns(3)
    with sl1:
        slippage_pct = st.slider(
            "Slippage per trade (%)", 0.0, 0.5, 0.10, 0.01,
            key="bt_slippage",
            help="Price impact on entry and exit. 0.10% is realistic for FTSE 100 / S&P 500.")
    with sl2:
        commission = st.number_input(
            "Commission per trade ($)", 0.0, 30.0, 5.0, 1.0,
            key="bt_commission",
            help="Fixed broker fee per trade. Use 0 for commission-free brokers.")
    with sl3:
        stop_slip_pct = st.slider(
            "Stop-loss slippage (%)", 0.0, 1.0, 0.20, 0.05,
            key="bt_stopslip",
            help="Extra slippage on stop-loss fills in fast markets. Usually 2× normal slippage.")

    if slippage_pct > 0 or commission > 0:
        st.info(f"Round-trip cost per trade: "
                f"**{slippage_pct*2:.2f}% slippage + ${commission*2:.0f} commission**. "
                f"On a $1,000 position that's ~${1000*slippage_pct*2 + commission*2:.0f} total.")

    @st.cache_data(ttl=600)
    def backtest_stat(ticker,start_str,capital,win_prob_key,
                      slip_pct,comm,stop_slip):
        all_tix=list(set([ticker,get_sector(ticker),"SPY","^VIX","^TNX","^FVX"]))
        raw=yf.download(all_tix,start=start_str,
                        end=datetime.today().strftime("%Y-%m-%d"),
                        auto_adjust=True,progress=False,group_by="ticker")
        c=get_col(raw,ticker,"Close");   v=get_col(raw,ticker,"Volume")
        vix=get_col(raw,"^VIX","Close"); spy=get_col(raw,"SPY","Close")
        sc=get_col(raw,get_sector(ticker),"Close")
        t10=get_col(raw,"^TNX","Close"); t5=get_col(raw,"^FVX","Close")
        if len(c)<60: return None,None,None,None
        idx=c.index
        vix=vix.reindex(idx,method="ffill"); spy=spy.reindex(idx,method="ffill")
        sc=sc.reindex(idx,method="ffill");   t10=t10.reindex(idx,method="ffill")
        t5=t5.reindex(idx,method="ffill")
        atr_s=(c.rolling(2).max()-c.rolling(2).min()).rolling(14).mean()
        cap=float(capital); pos=0; entry_px=0.0; stop_px=0.0; highest=0.0
        trades=[]; equity=[]; wp_list=[]
        total_costs=0.0

        for i in range(len(c)):
            price=c.iloc[i]; date=idx[i]
            atr=atr_s.iloc[i] if not pd.isna(atr_s.iloc[i]) else 0
            ci=c.iloc[:i+1]; vi=v.iloc[:i+1]
            vxi=vix.iloc[:i+1]; spi=spy.iloc[:i+1]
            sci=sc.iloc[:i+1]; t10i=t10.iloc[:i+1]; t5i=t5.iloc[:i+1]
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

            # ── Stop-loss with slippage ──
            if pos>0 and price<=stop_px:
                # Stop fills at stop price less extra stop slippage
                fill_px  = stop_px * (1 - stop_slip)
                cost     = comm + fill_px*pos*slip_pct
                proceeds = fill_px*pos - cost
                pnl      = proceeds - entry_px*pos
                cap     += proceeds
                total_costs += cost
                trades.append({
                    "Date":date,"Action":"STOP",
                    "Price":round(fill_px,2),"Shares":pos,
                    "PnL":round(pnl,2),"Costs":round(cost,2)})
                pos=0

            all_pass=(wp>=win_prob_key and trd and MIN_Z<=z<=MAX_Z and kel>0.01)

            if all_pass and pos==0:
                # ── Entry with slippage — pay ask (price + slippage) ──
                fill_px  = price * (1 + slip_pct)
                shares   = int(cap*kel/fill_px)
                if shares>0:
                    cost     = comm + fill_px*shares*slip_pct
                    cap     -= fill_px*shares + cost
                    pos      = shares
                    entry_px = fill_px
                    highest  = fill_px
                    stop_px  = fill_px - 1.5*atr
                    total_costs += cost
                    trades.append({
                        "Date":date,"Action":"BUY",
                        "Price":round(fill_px,2),"Shares":shares,
                        "PnL":0,"Costs":round(cost,2)})

            elif wp<0.45 and pos>0:
                # ── Exit with slippage — receive bid (price - slippage) ──
                fill_px  = price * (1 - slip_pct)
                cost     = comm + fill_px*pos*slip_pct
                proceeds = fill_px*pos - cost
                pnl      = proceeds - entry_px*pos
                cap     += proceeds
                total_costs += cost
                trades.append({
                    "Date":date,"Action":"SELL",
                    "Price":round(fill_px,2),"Shares":pos,
                    "PnL":round(pnl,2),"Costs":round(cost,2)})
                pos=0

            equity.append(cap+pos*price)

        if pos>0:
            fill_px  = c.iloc[-1]*(1-slip_pct)
            cost     = comm + fill_px*pos*slip_pct
            proceeds = fill_px*pos - cost
            pnl      = proceeds - entry_px*pos
            cap     += proceeds
            total_costs += cost
            trades.append({
                "Date":idx[-1],"Action":"CLOSE",
                "Price":round(fill_px,2),"Shares":pos,
                "PnL":round(pnl,2),"Costs":round(cost,2)})

        tdf = (pd.DataFrame(trades) if trades else
               pd.DataFrame(columns=["Date","Action","Price","Shares","PnL","Costs"]))
        return tdf, pd.Series(equity,index=idx), pd.Series(wp_list,index=idx), c, total_costs

    if st.button("▶️ Run backtest",type="primary",key="bt_run"):
        with st.spinner(f"Backtesting {bt_tick} — {bt_period} (with slippage)..."):
            bts=str(bt_start.date() if hasattr(bt_start,"date") else bt_start)
            result=backtest_stat(bt_tick,bts,bt_cap,MIN_WIN_PROB,
                                 slippage_pct/100,commission,stop_slip_pct/100)

        if result[0] is None:
            st.error("Not enough data. Try a wider period or different ticker.")
        else:
            tdf,eq_s,wp_s,cls_bt,total_costs=result
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

            m1,m2,m3,m4,m5,m6,m7=st.columns(7)
            m1.metric("Total return", f"{tr:.1f}%",delta=f"{tr-bh:+.1f}% vs B&H")
            m2.metric("Buy & hold",   f"{bh:.1f}%")
            m3.metric("Sharpe",       round(sh,2),
                      delta="Good ✅" if sh>=1 else "Weak",
                      delta_color="normal" if sh>=1 else "inverse")
            m4.metric("Max drawdown", f"{dd:.1f}%")
            m5.metric("Win rate",     f"{hr:.1f}%")
            m6.metric("Final value",  f"${eq_s.iloc[-1]:,.0f}")
            m7.metric("Total costs",  f"${total_costs:,.0f}",
                      delta=f"{total_costs/bt_cap*100:.1f}% of capital",
                      delta_color="inverse")

            st.markdown("---")

            fig_bt=go.Figure()
            fig_bt.add_trace(go.Scatter(x=cls_bt.index,y=cls_bt.values,
                                        name="Price",line=dict(color="#2196F3",width=1.5)))
            fig_bt.add_trace(go.Scatter(x=cls_bt.index,y=cls_bt.rolling(50).mean(),
                                        name="50MA",line=dict(color="#ff6e40",width=1,dash="dot")))
            if not tdf.empty:
                for act,sym,col in [("BUY","triangle-up","#00e676"),
                                     ("SELL","triangle-down","#ffd740"),
                                     ("STOP","x","#ff1744")]:
                    t=tdf[tdf["Action"]==act]
                    if len(t):
                        fig_bt.add_trace(go.Scatter(x=t["Date"],y=t["Price"],mode="markers",
                            marker=dict(symbol=sym,size=12,color=col),name=act))
            fig_bt.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                                 font_color="white",height=360,
                                 title=f"{bt_tick} — {bt_period} · {quality}")
            st.plotly_chart(fig_bt,use_container_width=True)

            bh_eq=(cls_bt/cls_bt.iloc[0])*bt_cap
            fig_eq=go.Figure()
            fig_eq.add_trace(go.Scatter(x=eq_s.index,y=eq_s.values,
                                        name="Model",line=dict(color="#00e676",width=2)))
            fig_eq.add_trace(go.Scatter(x=bh_eq.index,y=bh_eq.values,
                                        name="Buy & hold",
                                        line=dict(color="#90caf9",width=2,dash="dash")))
            fig_eq.add_hline(y=bt_cap,line_dash="dot",line_color="#555")
            fig_eq.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                                 font_color="white",height=280,
                                 yaxis_title="Portfolio value ($)")
            st.plotly_chart(fig_eq,use_container_width=True)

            fig_wp2=go.Figure()
            fig_wp2.add_trace(go.Scatter(x=wp_s.index,y=wp_s*100,fill="tozeroy",
                                         fillcolor="rgba(33,150,243,0.1)",
                                         line=dict(color="#2196F3",width=1)))
            fig_wp2.add_hline(y=MIN_WIN_PROB*100,line_dash="dash",line_color="#00e676",
                              annotation_text=f"Threshold {MIN_WIN_PROB*100:.0f}%")
            fig_wp2.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                                  font_color="white",height=180,
                                  yaxis=dict(range=[0,100],title="Win prob %"),
                                  title="Rolling Bayesian win probability")
            st.plotly_chart(fig_wp2,use_container_width=True)

            if not tdf.empty:
                st.markdown("#### Trade log")
                def ca(val):
                    c={"BUY":"#00e676","SELL":"#ffd740","STOP":"#ff1744",
                       "CLOSE":"#90caf9"}.get(val,"white")
                    return f"color:{c};font-weight:bold"
                def cp(val):
                    return f"color:{'#00e676' if val>0 else '#ff1744' if val<0 else 'white'}"
                st.dataframe(
                    tdf.style.map(ca,subset=["Action"]).map(cp,subset=["PnL"]),
                    use_container_width=True,height=240)
                st.download_button("⬇️ Download trades",tdf.to_csv(index=False),
                                   f"trades_{bt_tick}.csv","text/csv",
                                   key="bt_download")

# ── TAB 5: POSITIONS ──────────────────────────────────────────────────────────

with tab5:
    st.subheader("📂 Position Tracker")
    st.caption("Log open trades. The model monitors exit conditions and alerts you when to act.")

    if "positions" not in st.session_state:
        st.session_state.positions=[]

    st.markdown("#### Add a position")
    p1,p2,p3,p4,p5=st.columns(5)
    with p1:
        pos_ticker=st.text_input("Ticker",value="SHEL.L",key="pos_ticker_input").upper()
    with p2:
        pos_entry=st.number_input("Entry price",value=100.0,step=0.01,key="pos_entry")
    with p3:
        pos_shares=st.number_input("Shares",value=10,step=1,key="pos_shares")
    with p4:
        pos_stop=st.number_input("Stop-loss",value=90.0,step=0.01,key="pos_stop")
    with p5:
        pos_date=st.date_input("Entry date",value=datetime.today(),key="pos_date")
    pos_max_weeks=st.slider("Maximum hold (weeks)",2,16,8,key="pos_maxwks")

    if st.button("➕ Add position",type="primary",key="pos_add"):
        st.session_state.positions.append({
            "ticker":pos_ticker,"entry":pos_entry,"shares":pos_shares,
            "stop":pos_stop,"date":str(pos_date),"max_weeks":pos_max_weeks,
            "trail_high":pos_entry,
        })
        st.success(f"Added {pos_ticker} — {pos_shares} shares at ${pos_entry:.2f}")

    if not st.session_state.positions:
        st.info("No open positions yet. Add one above.")
    else:
        st.markdown("---")

        @st.cache_data(ttl=300)
        def fetch_pos_data(tickers_key):
            all_tix=list(set(list(tickers_key)+
                             list(SECTOR_MAP.values())+
                             list(FTSE_SECTOR_MAP.values())+
                             ["SPY","^VIX","^TNX","^FVX"]))
            end=datetime.today(); start=end-timedelta(days=420)
            return yf.download(all_tix,start=start,end=end,
                               auto_adjust=True,progress=False,group_by="ticker")

        pos_tix=tuple(set(p["ticker"] for p in st.session_state.positions))
        raw_pos=fetch_pos_data(pos_tix)
        vix_pos=get_col(raw_pos,"^VIX","Close"); spy_pos=get_col(raw_pos,"SPY","Close")
        t10_pos=get_col(raw_pos,"^TNX","Close"); t5_pos =get_col(raw_pos,"^FVX","Close")
        regime_pos,tradeable_pos,*_=classify_regime(vix_pos,spy_pos)
        to_remove=[]

        for idx_p,pos in enumerate(st.session_state.positions):
            ticker=pos["ticker"]
            c=get_col(raw_pos,ticker,"Close"); v=get_col(raw_pos,ticker,"Volume")
            sc=get_col(raw_pos,get_sector(ticker),"Close")
            if len(c)<60: st.warning(f"{ticker}: insufficient data."); continue

            cur=float(c.iloc[-1]); entry=pos["entry"]; shares=pos["shares"]
            stop=pos["stop"]
            entry_date=datetime.strptime(pos["date"],"%Y-%m-%d")
            days_held=(datetime.today()-entry_date).days
            weeks_held=days_held/7; max_wks=pos["max_weeks"]
            pnl_pct=(cur-entry)/entry*100; pnl_cash=(cur-entry)*shares
            atr=float((c.rolling(2).max()-c.rolling(2).min()).rolling(14).mean().iloc[-1])
            trail_high=max(pos.get("trail_high",entry),cur)
            st.session_state.positions[idx_p]["trail_high"]=trail_high
            trail_stop=trail_high-2.0*atr

            try:
                dms=compute_dms(c,v,vix_pos,spy_pos,sc,t10_pos,t5_pos)
                wp,_=bayesian_win_prob(c,dms)
                z=zscore_entry(c)
            except:
                dms=50; wp=0.5; z=0.0

            exit_urgent=[]; exit_signals=[]; profit_flags=[]
            if cur<=stop:           exit_urgent.append("🔴 STOP-LOSS HIT — exit immediately")
            if cur<=trail_stop:     exit_urgent.append("🔴 TRAILING STOP HIT — exit immediately")
            if wp<0.45:             exit_signals.append(f"⚠️ Win prob fell to {wp*100:.1f}%")
            if dms<40:              exit_signals.append(f"⚠️ DMS fell to {dms:.0f}")
            if z>3.0:               exit_signals.append(f"⚠️ Overbought — Z {z:.2f}")
            if not tradeable_pos:   exit_signals.append(f"⚠️ Regime: {regime_pos}")
            if weeks_held>=max_wks: exit_signals.append(f"⚠️ Max hold {max_wks}wks reached")
            if pnl_cash>=3*atr*shares: profit_flags.append("💰 +3× ATR gain — consider selling ⅓")
            if pnl_cash>=2*atr*shares: profit_flags.append("💰 +2× ATR gain — consider selling ⅓")
            if weeks_held>=2 and pnl_pct>0:
                profit_flags.append(f"✅ Move stop to breakeven (${entry:.2f})")

            if exit_urgent:     sc_="⛔ EXIT NOW"; bc_="#ff1744"
            elif exit_signals:  sc_="⚠️ REVIEW";   bc_="#ff6e40"
            elif profit_flags:  sc_="💰 MANAGE";   bc_="#ffd740"
            else:               sc_="✅ HOLD";      bc_="#00e676"

            st.markdown(
                f'<div style="background:#1c2030;border-left:4px solid {bc_};'
                f'border-radius:8px;padding:14px 18px;margin-bottom:6px;">'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="color:white;font-size:18px;font-weight:700">{ticker}</span>'
                f'<span style="color:{bc_};font-weight:700">{sc_}</span></div>'
                f'<div style="color:#888;font-size:12px">'
                f'{shares} shares · entered ${entry:.2f} · {pos["date"]}</div></div>',
                unsafe_allow_html=True)

            m1,m2,m3,m4,m5,m6,m7=st.columns(7)
            m1.metric("Price",     f"${cur:.2f}")
            m2.metric("P&L",       f"${pnl_cash:+,.0f}",delta=f"{pnl_pct:+.1f}%",
                      delta_color="normal" if pnl_cash>=0 else "inverse")
            m3.metric("Value",     f"${cur*shares:,.0f}")
            m4.metric("Held",      f"{days_held}d / {max_wks}wk max")
            m5.metric("Win prob",  f"{wp*100:.1f}%",
                      delta="✅" if wp>=0.55 else "❌",
                      delta_color="normal" if wp>=0.55 else "inverse")
            m6.metric("DMS",       f"{dms:.0f}",
                      delta="✅" if dms>=40 else "❌",
                      delta_color="normal" if dms>=40 else "inverse")
            m7.metric("Trail stop",f"${trail_stop:.2f}")

            for msg in exit_urgent:  st.error(msg)
            for msg in exit_signals: st.warning(msg)
            if profit_flags and not exit_urgent:
                for msg in profit_flags: st.info(msg)
            if not exit_urgent and not exit_signals and not profit_flags:
                st.success("All conditions healthy — hold.")

            with st.expander(f"📈 {ticker} chart"):
                fig_pos=go.Figure()
                fig_pos.add_trace(go.Scatter(x=c.index,y=c.values,name="Price",
                                             line=dict(color="#2196F3",width=1.5)))
                fig_pos.add_trace(go.Scatter(x=c.index,y=c.rolling(20).mean(),name="20MA",
                                             line=dict(color="#ffd740",width=1,dash="dash")))
                fig_pos.add_hline(y=entry,line_dash="dot",line_color="#90caf9",
                                  annotation_text=f"Entry ${entry:.2f}",
                                  annotation_font_color="#90caf9")
                fig_pos.add_hline(y=stop,line_dash="dash",line_color="#ff1744",
                                  annotation_text=f"Stop ${stop:.2f}",
                                  annotation_font_color="#ff1744")
                fig_pos.add_hline(y=trail_stop,line_dash="dot",line_color="#ff6e40",
                                  annotation_text=f"Trail ${trail_stop:.2f}",
                                  annotation_font_color="#ff6e40")
                fig_pos.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1c2030",
                                      font_color="white",height=280,
                                      margin=dict(l=0,r=0,t=20,b=0))
                st.plotly_chart(fig_pos,use_container_width=True)

            if st.button(f"🗑️ Remove {ticker}",key=f"rm_{idx_p}_{ticker}"):
                to_remove.append(idx_p)
            st.markdown("")

        for i in sorted(to_remove,reverse=True):
            st.session_state.positions.pop(i)
        if to_remove: st.rerun()

        if len(st.session_state.positions)>1:
            st.markdown("---")
            tot_pnl=sum(
                (get_col(raw_pos,p["ticker"],"Close").iloc[-1]-p["entry"])*p["shares"]
                for p in st.session_state.positions
                if len(get_col(raw_pos,p["ticker"],"Close"))>0)
            tot_val=sum(
                get_col(raw_pos,p["ticker"],"Close").iloc[-1]*p["shares"]
                for p in st.session_state.positions
                if len(get_col(raw_pos,p["ticker"],"Close"))>0)
            ps1,ps2,ps3=st.columns(3)
            ps1.metric("Open positions",len(st.session_state.positions))
            ps2.metric("Total P&L",     f"${tot_pnl:+,.0f}",
                       delta_color="normal" if tot_pnl>=0 else "inverse")
            ps3.metric("Total value",   f"${tot_val:,.0f}")

st.markdown("---")
st.caption("⚠️ For informational purposes only. Not financial advice.")
