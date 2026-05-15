"""
Equity Quant Dashboard — Weekly 3-Factor Model
Variables: Price momentum, Volume, Volatility
Check weekly, hold weeks to months
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

# ── FIXED PARAMETERS ──────────────────────────────────────────────────────────

KELLY_FRAC     = 0.30   # 30% Kelly — Sharpe optimised
MAX_POS        = 0.12   # 12% max per position
MC_N           = 1000   # Monte Carlo simulations
REGIME_VIX     = 25     # VIX ceiling for bull regime
FORWARD_DAYS   = 20     # forward return window — matches 4-week hold
MIN_Z          = 0.3    # loose Z floor — win prob does the filtering
MAX_Z          = 3.0    # loose Z ceiling

# 3-factor weights — price, volume, volatility only
WEIGHTS = {
    "price_momentum": 0.45,   # strongest predictor for weekly holds
    "volume":         0.30,   # conviction filter
    "volatility":     0.25,   # risk-adjusted entry
}

# ── UNIVERSES ─────────────────────────────────────────────────────────────────

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

FTSE250 = [
    "3IN.L","AEW.L","AGR.L","AJB.L","ANIC.L","APAX.L","ATG.L","ATM.L",
    "ATST.L","AWE.L","BBY.L","BCPT.L","BME.L","BMS.L","BNKR.L","BOO.L",
    "BOWL.L","BRW.L","BYG.L","CAL.L","CBOX.L","CEY.L","CHG.L","CLDN.L",
    "CMCX.L","CNE.L","CTEC.L","CVS.L","CWK.L","DARK.L","DCG.L","DKG.L",
    "DLG.L","DOCS.L","DRX.L","DSG.L","DWF.L","EMG.L","ERM.L","ESNT.L",
    "EWI.L","EZJ.L","FCIT.L","FGP.L","FLT.L","FRAS.L","GAW.L","GBG.L",
    "GEM.L","GENL.L","GLE.L","GMG.L","GNS.L","GRG.L","GSS.L","GYM.L",
    "HAS.L","HICL.L","HIG.L","HOME.L","HRI.L","HSL.L","HWDN.L","IDS.L",
    "IGR.L","IMI.L","INPP.L","INVP.L","IP.L","IQG.L","JET2.L","JMG.L",
    "JPE.L","KBC.L","KGF.L","KWS.L","LAM.L","LBG.L","LDG.L","LGB.L",
    "LMP.L","LSL.L","LXI.L","MAB.L","MCR.L","MEGA.L","MERC.L","MHN.L",
    "MILS.L","MLPE.L","MMB.L","MNKS.L","MOB.L","MRC.L","MRL.L","MSI.L",
    "MTC.L","MTO.L","MTW.L","NAH.L","NCC.L","NCYF.L","NTG.L","OCI.L",
    "OCN.L","OML.L","ONT.L","PAGE.L","PCA.L","PCTN.L","PDG.L","PEG.L",
    "PFG.L","PGIT.L","PHI.L","PIC.L","PLND.L","PLG.L","PMP.L","POG.L",
    "PPH.L","PRI.L","PRSR.L","PSH.L","PTEC.L","QQ.L","RCP.L","RDW.L",
    "RGU.L","RMG.L","ROR.L","RPS.L","SAFE.L","SDY.L","SEE.L","SFR.L",
    "SHP.L","SIR.L","SKG.L","SLPE.L","SMIF.L","SND.L","SNWS.L","SOM.L",
    "SPH.L","SPI.L","SQZ.L","STOB.L","STS.L","SUP.L","SUPR.L","SVE.L",
    "SXS.L","TCAP.L","TCG.L","TET.L","TIG.L","TLW.L","TMPL.L","TRI.L",
    "TRN.L","TRS.L","TWD.L","UCG.L","UEM.L","UTG.L","VCT.L","VSVS.L",
    "VTU.L","WEIR.L","WHR.L","WIX.L","WKP.L","WMH.L","XAR.L","XPS.L",
    "YCA.L","ZTF.L",
]

FTSE350 = list(dict.fromkeys(FTSE100 + FTSE250))

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
    "JET2.L":"XLY","EZJ.L":"XLY","BOO.L":"XLY","FRAS.L":"XLY","GYM.L":"XLY",
    "GAW.L":"XLY","HOME.L":"XLY","DOCS.L":"XLY","MAB.L":"XLY",
    "IMI.L":"XLI","WEIR.L":"XLI","MTO.L":"XLI","TRN.L":"XLI","ROR.L":"XLI",
    "SXS.L":"XLI","HWDN.L":"XLI","ATM.L":"XLI","AWE.L":"XLI",
    "CHG.L":"XLV","CVS.L":"XLV","SHP.L":"XLV","HRI.L":"XLV",
    "CMCX.L":"XLC","IDS.L":"XLC","WMH.L":"XLC","PAGE.L":"XLC","QQ.L":"XLC",
    "DLG.L":"XLF","LBG.L":"XLF","AJB.L":"XLF","PFG.L":"XLF","INVP.L":"XLF",
    "TCAP.L":"XLF","JMG.L":"XLF","MCR.L":"XLF","LSL.L":"XLF","PRI.L":"XLF",
    "POG.L":"XLB","CEY.L":"XLB","PEG.L":"XLB","SIR.L":"XLB",
    "CNE.L":"XLE","TLW.L":"XLE","EMG.L":"XLE",
    "GBG.L":"XLK","DARK.L":"XLK","CTEC.L":"XLK","ESNT.L":"XLK","NCC.L":"XLK",
    "PTEC.L":"XLK","DSG.L":"XLK","EWI.L":"XLK",
    "SGRO.L":"XLRE","LXI.L":"XLRE","SUPR.L":"XLRE","PRSR.L":"XLRE",
    "HICL.L":"XLU","INPP.L":"XLU","TRIG.L":"XLU","NCYF.L":"XLU",
    "GRG.L":"XLP","SNWS.L":"XLP","PMP.L":"XLP",
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
    "XOM":"XLE","V":"XLF","AVGO":"XLK","PG":"XLP","MA":"XLF","JNH":"XLV",
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
    "JNJ":"XLV",
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
        key="sidebar_watchlist")
    WATCHLIST = [t.strip().upper() for t in user_input.split(",") if t.strip()]

    st.markdown("---")
    st.markdown("### Signal quality")
    quality = st.select_slider(
        "Win probability required",
        options=["Moderate (≥55%)", "Strong (≥60%)", "Very Strong (≥70%)"],
        value="Strong (≥60%)")
    MIN_WIN_PROB = {
        "Moderate (≥55%)":0.55,
        "Strong (≥60%)":0.60,
        "Very Strong (≥70%)":0.70}[quality]

    st.markdown("---")
    with st.expander("ℹ️ Model details"):
        st.caption(
            "**3 variables only:**\n\n"
            "📈 **Price momentum** (45%) — trend direction and strength\n"
            "📊 **Volume** (30%) — conviction behind the move\n"
            "🌊 **Volatility** (25%) — risk-adjusted entry timing\n\n"
            "**Forward return window:** 20 days (4 weeks)\n"
            "Matches your weekly check / 4–8 week hold.\n\n"
            "**Kelly fraction:** 30% — Sharpe optimised\n"
            "**Check cadence:** Weekly"
        )
    st.button("🔄 Refresh", use_container_width=True, type="primary")
    st.caption(f"Week of: {datetime.now().strftime('%d %b %Y')}")

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

# ── 3-FACTOR SCORE ────────────────────────────────────────────────────────────

def compute_score(c, v, vix_s):
    """
    3-factor score using price momentum, volume and volatility only.
    Returns score 0-100 and individual factor scores.
    """
    # ── Factor 1: Price momentum (45%) ──
    # 20-day return normalised — captures medium-term trend
    ret_20  = (c.iloc[-1] / c.iloc[-20]) - 1 if len(c) >= 20 else 0
    # 5-day return — captures recent acceleration
    ret_5   = (c.iloc[-1] / c.iloc[-5])  - 1 if len(c) >= 5  else 0
    # Z-score of today's return vs 20-day distribution
    rets    = c.pct_change().dropna()
    mu      = rets.rolling(20).mean().iloc[-1]
    sd      = rets.rolling(20).std().iloc[-1]
    z       = (rets.iloc[-1] - mu) / sd if sd > 0 else 0
    z_norm  = normalise(z, -3, 3)
    # Combine: 50% medium-term, 30% recent, 20% z-score
    pm_raw  = 0.50*normalise(ret_20,-0.15,0.15) + \
              0.30*normalise(ret_5, -0.05,0.05)  + \
              0.20*z_norm
    price_score = pm_raw

    # ── Factor 2: Volume (30%) ──
    # Rising volume on up days = strong conviction
    avg_vol    = v.rolling(20).mean().iloc[-1]
    vol_ratio  = v.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
    # Price direction × volume ratio
    direction  = 1 if rets.iloc[-1] > 0 else -1
    vol_signal = direction * min(vol_ratio, 3.0)  # cap at 3× average
    vol_score  = normalise(vol_signal, -3, 3)

    # ── Factor 3: Volatility (25%) ──
    # Low VIX + low ATR = calm entry = better Sharpe
    vix_score  = 100 - normalise(vix_s.iloc[-1], 10, 40)
    hi = c.rolling(2).max(); lo = c.rolling(2).min()
    atr_pct    = ((hi-lo)/c).rolling(14).mean().iloc[-1] * 100
    atr_score  = 100 - normalise(atr_pct, 0.3, 4.0)
    vol_f_score= (vix_score + atr_score) / 2

    # ── Combined score ──
    total = (WEIGHTS["price_momentum"] * price_score +
             WEIGHTS["volume"]         * vol_score   +
             WEIGHTS["volatility"]     * vol_f_score)

    return round(total, 2), round(price_score,1), round(vol_score,1), round(vol_f_score,1), round(z,2)

# ── STATISTICAL LAYERS ────────────────────────────────────────────────────────

def bayesian_win_prob(c, score, window=20, horizon=20):
    """
    P(price up over next horizon days) given similar score historically.
    horizon=20 matches a 4-week holding period (weekly checks).
    """
    rets  = c.pct_change().dropna()
    rm    = rets.rolling(window).mean()
    rs    = rets.rolling(window).std()
    # Proxy score from historical data
    proxy = ((rm/rs.replace(0,np.nan)).clip(-3,3)*16.67+50).dropna()
    fwd   = c.pct_change(horizon).shift(-horizon)
    aln   = pd.concat([proxy,fwd],axis=1).dropna()
    aln.columns=["s","f"]
    lo=max(0,score-10); hi=min(100,score+10)
    b = aln[(aln["s"]>=lo)&(aln["s"]<=hi)]
    if len(b)<10: return 0.50, len(b)
    return round(float((b["f"]>0).mean()),3), len(b)

def classify_regime(vix_s, spy_s):
    vix=vix_s.iloc[-1]; spy=spy_s.iloc[-1]
    spy200=spy_s.rolling(200).mean().iloc[-1]; above=spy>spy200
    if above and vix<15:          return "Bull quiet",    True,  2.1, vix, spy, spy200
    if above and vix<REGIME_VIX:  return "Bull volatile", True,  0.8, vix, spy, spy200
    if not above and vix<20:      return "Bear quiet",    False,-0.3, vix, spy, spy200
    return "Bear volatile", False, -2.4, vix, spy, spy200

def kelly_size(win_prob, avg_win=0.025, avg_loss=0.012):
    if avg_loss==0: return 0.0
    b=avg_win/avg_loss; q=1-win_prob
    return round(min(max((b*win_prob-q)/b*KELLY_FRAC,0),MAX_POS),4)

def monte_carlo(c, kelly, capital=100000, horizon=20):
    ret   = c.pct_change().dropna().values
    entry = c.iloc[-1]; s