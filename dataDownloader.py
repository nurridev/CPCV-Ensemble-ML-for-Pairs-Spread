from __future__ import annotations

"""Download price + fundamental data from Alpha Vantage and
save **one CSV per stock**, containing **all available columns**.

Key updates vs. the original script
-----------------------------------
* **Single‑file output per ticker** → files are now named ``<TICKER>.csv``.
* **All price columns kept** (Open, High, Low, Close, Adj Close, Volume …).
* **Fundamental fields merged** into the same file (one column per field).
* **Fast**: still uses a thread‑pool + token‑bucket throttling.
"""

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, List, Any

import pandas as pd
import requests
from tqdm import tqdm

# ────────── ENV / USER CONFIG ──────────
API_KEY_PRICE   = os.getenv("ALPHAVANTAGE_KEY", "JFEOWIO6WDQBH214")
PRICE_RPM       = int(os.getenv("ALPHAVANTAGE_RPM", "5"))

API_KEY_FUND    = os.getenv("ALPHAVANTAGE_FUND_KEY", API_KEY_PRICE)
FUND_RPM        = int(os.getenv("ALPHAVANTAGE_FUND_RPM", str(PRICE_RPM)))

START_DATE      = os.getenv("START_DATE", "2010-01-01")
END_DATE        = os.getenv("END_DATE", "2023-01-01")
GRANULARITY     = os.getenv("GRANULARITY", "daily")  # daily / weekly / monthly / 1min / 5min / …

DOWNLOAD_FUND   = bool(int(os.getenv("ALPHAVANTAGE_FUNDAMENTALS", "1")))
SAVE_COMBINED   = bool(int(os.getenv("SAVE_COMBINED", "0")))  # optional: all tickers → one wide CSV

WORKERS         = int(os.getenv("ALPHAVANTAGE_WORKERS", str(min(32, PRICE_RPM))))

TICKERS: List[str] = [
    # Full Russell 1000 Constituents (as listed)
    'TXG', 'MMM', 'AOS', 'AAON', 'ABT', 'ABBV', 'ACHC', 'ACN', 'AYI', 'ADBE', 'ADT', 'AAP', 'WMS', 'AMD', 'ACM', 'AES',
    'AMG', 'AFRM', 'AFL', 'AGCO', 'A', 'ADC', 'AGNC', 'AL', 'APD', 'ABNB', 'AKAM', 'ALK', 'ALB', 'ACI', 'AA', 'ARE',
    'ALGN', 'ALLE', 'ALGM', 'LNT', 'ALSN', 'ALL', 'ALLY', 'ALNY', 'GOOGL', 'GOOG', 'MO', 'AMCR', 'DOX', 'AMED', 'AMTM',
    'AS', 'AEE', 'AAL', 'AEP', 'AXP', 'AFG', 'AMH', 'AIG', 'AMT', 'AWK', 'COLD', 'AMP', 'AME', 'AMGN', 'AMKR', 'APH',
    'ADI', 'NLY', 'ANSS', 'AM', 'AR', 'AON', 'APA', 'APG', 'APLS', 'APO', 'APPF', 'AMAT', 'APP', 'ATR', 'APTV',
    'ARMK', 'ACGL', 'ADM', 'ARES', 'ANET', 'AWI', 'ARW', 'AJG', 'ASH', 'AZPN', 'AIZ', 'AGO', 'ALAB', 'T', 'ATI', 'TEAM',
    'ATO', 'ADSK', 'ADP', 'AN', 'AZO', 'AVB', 'AGR', 'AVTR', 'AVY', 'CAR', 'AVT', 'AXTA', 'AXS', 'AXON', 'AZEK', 'AZTA',
    'BKR', 'BALL', 'BAC', 'OZK', 'BBWI', 'BAX', 'BDX', 'BRBR', 'BSY', 'BRK.B', 'BERY', 'BBY', 'BILL', 'BIO', 'TECH',
    'BIIB', 'BMRN', 'BIRK', 'BJ', 'BLK', 'BX', 'HRB', 'OWL', 'BK', 'BA', 'BOKF', 'BKNG', 'BAH', 'BWA', 'SAM', 'BSX',
    'BYD', 'BFAM', 'BHF', 'BMY', 'BRX', 'AVGO', 'BR', 'BEPC', 'BRO', 'BRKR', 'BC', 'BLDR', 'BG', 'BURL',
    'BWXT', 'BXP', 'CHRW', 'CACI', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CPRI', 'CAH', 'CSL', 'CG', 'KMX', 'CCL', 'CARR',
    'CRI', 'CVNA', 'CASY', 'CTLT', 'CAT', 'CAVA', 'CBOE', 'CBRE', 'CCCS', 'CDW', 'CE', 'CELH', 'COR', 'CNC', 'CNP',
    'CERT', 'CF', 'CRL', 'SCHW', 'CHTR', 'CHE', 'CC', 'LNG', 'CVX', 'CMG', 'CHH', 'CHRD', 'CB', 'CHD', 'CHDN', 'CIEN',
    'CI', 'CINF', 'CTAS', 'CRUS', 'CSCO', 'C', 'CFG', 'CIVI', 'CLVT', 'CLH', 'CWEN', 'CLF', 'CLX', 'NET', 'CME',
    'CMS', 'CNA', 'CNH', 'KO', 'COKE', 'CGNX', 'CTSH', 'COHR', 'COIN', 'CL', 'COLB', 'COLM', 'CMCSA', 'CMA', 'FIX',
    'CBSH', 'CAG', 'CNXC', 'CFLT', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'CNM', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST',
    'CTRA', 'COTY', 'CPNG', 'CUZ', 'CR', 'CXT', 'CACC', 'CRH', 'CROX', 'CRWD', 'CCI', 'CCK', 'CSX', 'CUBE', 'CMI', 'CW',
    'CVS', 'DHI', 'DHR', 'DRI', 'DAR', 'DDOG', 'DVA', 'DAY', 'DECK', 'DE', 'DAL', 'DELL', 'XRAY', 'DVN', 'DXCM', 'FANG',
    'DKS', 'DLR', 'DDS', 'DFS', 'DOCU', 'DLB', 'DG', 'DLTR', 'D', 'DPZ', 'DCI', 'DASH', 'DV', 'DOV', 'DOW', 'DOCS', 'DKNG',
    'DBX', 'DTM', 'DTE', 'DUK', 'DNB', 'DUOL', 'DD', 'BROS', 'DXC', 'DT', 'ELF', 'EXP', 'EWBC', 'EGP', 'EMN', 'ETN', 'EBAY',
    'ECL', 'EIX', 'EW', 'ELAN', 'ESTC', 'EA', 'ESI', 'ELV', 'EME', 'EMR', 'EHC', 'ENOV', 'ENPH', 'ENTG', 'ETR', 'NVST',
    'EOG', 'EPAM', 'EPR', 'EQT', 'EFX', 'EQIX', 'EQH', 'ELS', 'EQR', 'ESAB', 'WTRG', 'ESS', 'EL', 'ETSY', 'EEFT', 'EVR',
    'EG', 'EVRG', 'ES', 'ECG', 'EXAS', 'EXEL', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST',
    'FRT', 'FDX', 'FERG', 'FNF', 'FIS', 'FITB', 'FAF', 'FCNCA', 'FHB', 'FHN', 'FR', 'FSLR', 'FE', 'FI', 'FIVE', 'FIVN',
    'FND', 'FLO', 'FLS', 'FMC', 'FNB', 'F', 'FTNT', 'FTV', 'FTRE', 'FBIN', 'FOXA', 'FOX', 'BEN', 'FCX', 'FRPT', 'FYBR',
    'CFR', 'FCN', 'GME', 'GLPI', 'GAP', 'GRMN', 'IT', 'GTES', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'G',
    'GNTX', 'GPC', 'GILD', 'GTLB', 'GPN', 'GFS', 'GLOB', 'GL', 'GMED', 'GDDY', 'GS', 'GGG', 'GRAL', 'LOPE', 'GPK', 'GO',
    'GWRE', 'GXO', 'HAL', 'THG', 'HOG', 'HIG', 'HAS', 'HCP', 'HAYW', 'HCA', 'HR', 'DOC', 'HEI', 'JKHY', 'HSY',
    'HES', 'HPE', 'HXL', 'DINO', 'HIW', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HLI', 'HHH', 'HWM', 'HPQ', 'HUBB', 'HUBS',
    'HUM', 'HBAN', 'HII', 'HUN', 'H', 'IAC', 'IBM', 'IDA', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY', 'INFA', 'IR', 'INGR',
    'INSP', 'PODD', 'INTC', 'IBKR', 'ICE', 'IFF', 'IP', 'IPG', 'ITCI', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IONS', 'IPGP', 'IQV',
    'IRDM', 'IRM', 'ITT', 'JBL', 'J', 'JHG', 'JAZZ', 'JBHT', 'JEF', 'JNJ', 'JCI', 'JLL', 'JPM', 'JNPR', 'KBR', 'K', 'KMPR',
    'KVUE', 'KDP', 'KEY', 'KEYS', 'KRC', 'KMB', 'KIM', 'KMI', 'KNSL', 'KEX', 'KKR', 'KLAC', 'KNX', 'KSS', 'KHC', 'KR', 'KD',
    'LHX', 'LH', 'LRCX', 'LAMR', 'LW', 'LSTR', 'LVS', 'LSCC', 'LAZ', 'LEA', 'LEG', 'LDOS', 'LEN', 'LII', 'LBRDA',
    'LBRDK', 'LBTYA', 'LBTYK', 'FWONA', 'FWONK', 'LLYVA', 'LLYVK', 'LNW', 'LLY', 'LECO', 'LNC', 'LIN', 'LINE', 'LAD', 'LFUS',
    'LYV', 'LKQ', 'LOAR', 'LMT', 'L', 'LPX', 'LOW', 'LPLA', 'LCID', 'LULU', 'LITE', 'LYFT', 'LYB', 'MTB', 'MTSI', 'M', 'MSGS',
    'MANH', 'MAN', 'CART', 'MPC', 'MKL', 'MKTX', 'MAR', 'VAC', 'MMC', 'MLM', 'MRVL', 'MAS', 'MASI', 'MTZ', 'MA', 'MTDR',
    'MTCH', 'MAT', 'MKC', 'MCD', 'MCK', 'MDU', 'MPW', 'MEDP', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MTG', 'MGM', 'MCHP', 'MU',
    'MSFT', 'MSTR', 'MAA', 'MIDD', 'MKSI', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MDB', 'MPWR', 'MNST', 'MCO', 'MS', 'MORN',
    'MOS', 'MSI', 'MP', 'MSA', 'MSM', 'MSCI', 'MUSA', 'NDAQ', 'NTRA', 'NFG', 'NSA', 'NCNO', 'NTAP', 'NFLX', 'NBIX', 'NFE',
    'NYT', 'NWL', 'NEU', 'NEM', 'NWSA', 'NWS', 'NXST', 'NEE', 'NKE', 'NI', 'NNN', 'NDSN', 'JWN', 'NSC', 'NTRS', 'NOC', 'NCLH',
    'NOV', 'NRG', 'NU', 'NUE', 'NTNX', 'NVT', 'NVDA', 'NVR', 'ORLY', 'OXY', 'OGE', 'OKTA', 'ODFL', 'ORI', 'OLN', 'OLLI',
    'OHI', 'OMC', 'ON', 'OMF', 'OKE', 'ONTO', 'ORCL', 'OGN', 'OSK', 'OTIS', 'OVV', 'OC', 'PCAR', 'PKG', 'PLTR', 'PANW',
    'PARAA', 'PARA', 'PK', 'PH', 'PSN', 'PAYX', 'PAYC', 'PYCR', 'PCTY', 'PYPL', 'PEGA', 'PENN', 'PAG', 'PNR', 'PEN', 'PEP',
    'PFGC', 'PR', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PPC', 'PNFP', 'PNW', 'PINS', 'PLNT', 'PLTK', 'PNC', 'PII', 'POOL',
    'BPOP', 'POST', 'PPG', 'PPL', 'PINC', 'TROW', 'PRI', 'PFG', 'PCOR', 'PG', 'PGR', 'PLD', 'PB', 'PRU', 'PTC', 'PSA', 'PEG',
    'PHM', 'PSTG', 'PVH', 'QGEN', 'QRVO', 'QCOM', 'PWR', 'QS', 'DGX', 'QDEL', 'RL', 'RRC', 'RJF', 'RYN', 'RBA', 'RBC', 'O',
    'RRX', 'REG', 'REGN', 'RF', 'RGA', 'RS', 'RNR', 'RGEN', 'RSG', 'RMD', 'RVTY', 'REXR', 'REYN', 'RH', 'RNG', 'RITM', 'RIVN',
    'RLI', 'RHI', 'HOOD', 'RBLX', 'RKT', 'ROK', 'ROIV', 'ROKU', 'ROL', 'ROP', 'ROST', 'RCL', 'RGLD', 'RPRX', 'RPM', 'RTX',
    'RYAN', 'R', 'SPGI', 'SAIA', 'SAIC', 'CRM', 'SLM', 'SRPT', 'SBAC', 'HSIC', 'SLB', 'SNDR', 'SMG', 'SEB', 'SEE', 'SEG',
    'SEIC', 'SRE', 'ST', 'S', 'SCI', 'NOW', 'SN', 'FOUR', 'SLGN', 'SPG', 'SSD', 'SIRI', 'SITE', 'SKX', 'SWKS', 'SMAR',
    'SJM', 'SW', 'SNA', 'SNOW', 'SOFI', 'SOLV', 'SON', 'SHC', 'SO', 'SCCO', 'LUV', 'SPB', 'SPR', 'SPOT', 'SSNC', 'STAG',
    'SWK', 'SBUX', 'STWD', 'STT', 'STLD', 'STE', 'SF', 'SYK', 'SUI', 'SMCI', 'SYF', 'SNPS', 'SNV', 'SYY', 'TMUS', 'TTWO',
    'TPR', 'TRGP', 'TGT', 'SNX', 'FTI', 'TDY', 'TFX', 'THC', 'TDC', 'TER', 'TSLA', 'TTEK', 'TXN', 'TPL', 'TXRH', 'TXT',
    'TMO', 'TFSL', 'THO', 'TKR', 'TJX', 'TKO', 'TOST', 'TOL', 'BLD', 'TTC', 'TPG', 'TSCO', 'TTD', 'TW', 'TT', 'TDG', 'TRU',
    'TNL', 'TRV', 'TREX', 'TRMB', 'TRIP', 'TFC', 'DJT', 'TWLO', 'TYL', 'TSN', 'UHAL', 'USB', 'X', 'UBER', 'UI',
    'UDR', 'UGI', 'PATH', 'ULTA', 'RARE', 'UAA', 'UA', 'UNP', 'UAL', 'UPS', 'URI', 'UTHR', 'UWMC', 'UNH', 'U', 'OLED', 'UHS',
    'UNM', 'USFD', 'MTN', 'VLO', 'VMI', 'VVV', 'VEEV', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VRT', 'VSTS', 'VFC',
    'VTRS', 'VICI', 'VKTX', 'VNOM', 'VIRT', 'V', 'VST', 'VNT', 'VNO', 'VOYA', 'VMC', 'WPC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT',
    'DIS', 'WBD', 'WM', 'WAT', 'WSO', 'W', 'WFRD', 'WBS', 'WEC', 'WFC', 'WELL', 'WEN', 'WCC', 'WST', 'WAL', 'WDC', 'WU', 'WLK',
    'WEX', 'WY', 'WHR', 'WTM', 'WMB', 'WSM', 'WTW', 'WSC', 'WING', 'WTFC', 'WOLF', 'WWD', 'WDAY', 'WH', 'WYNN', 'XEL', 'XP',
    'XPO', 'XYL', 'YETI', 'YUM', 'ZBRA', 'ZG', 'Z', 'ZBH', 'ZION', 'ZTS', 'ZM', 'ZI', 'ZS'
   ]

OUT_DIR = Path("stock_data")
OUT_DIR.mkdir(exist_ok=True)
# ───────────────────────────────────────

BASE_URL = "https://www.alphavantage.co/query"

###############################################################################
#   Token‑bucket session                                                      #
###############################################################################
class ThrottledSession:
    """Simple token‑bucket rate limiter around ``requests.Session`` (thread‑safe)."""

    def __init__(self, rpm: int):
        self._rpm = rpm
        self._tokens = rpm
        self._last = time.time()
        self._lock = threading.Lock()
        self._sess = requests.Session()

    def _acquire(self):
        while True:
            with self._lock:
                now = time.time()
                refill = (now - self._last) * self._rpm / 60
                if refill >= 1:
                    self._tokens = min(self._rpm, self._tokens + int(refill))
                    self._last = now
                if self._tokens:
                    self._tokens -= 1
                    return
            time.sleep(0.02)

    def get(self, *a, **kw):
        self._acquire()
        return self._sess.get(*a, **kw)


price_http = ThrottledSession(PRICE_RPM)
fund_http  = ThrottledSession(FUND_RPM)

###############################################################################
#   Helpers                                                                   #
###############################################################################

_DEF_TIME_SERIES: Dict[str, str] = {
    "daily": "TIME_SERIES_DAILY_ADJUSTED",
    "weekly": "TIME_SERIES_WEEKLY_ADJUSTED",
    "monthly": "TIME_SERIES_MONTHLY_ADJUSTED",
}

_INTRADAY_SET = {"1min", "5min", "15min", "30min", "60min"}


def _function_and_params() -> tuple[str, Dict[str, str]]:
    """Return AV function name + extra params for chosen granularity."""
    if GRANULARITY in _DEF_TIME_SERIES:
        return _DEF_TIME_SERIES[GRANULARITY], {}
    if GRANULARITY in _INTRADAY_SET:
        return "TIME_SERIES_INTRADAY", {"interval": GRANULARITY}
    raise ValueError(f"Unsupported GRANULARITY '{GRANULARITY}'")


def _call_av(http: ThrottledSession, params: Dict[str, str]) -> Dict[str, Any]:
    """Resilient 3‑try wrapper around Alpha Vantage JSON endpoints."""
    for _ in range(3):
        try:
            r = http.get(BASE_URL, params=params, timeout=10)
            r.raise_for_status()
            j = r.json()
            bad = {"Note", "Information", "Error Message"} & j.keys()
            if bad:
                raise RuntimeError(next(iter(bad)) + ": " + next(iter(j.values())))
            return j
        except Exception:  # noqa: BLE001,E722 – simple retry is fine
            time.sleep(2)
    raise RuntimeError("3 retries failed")


###############################################################################
#   Per‑ticker logic                                                          #
###############################################################################

def _clean_price_df(raw: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Convert raw AV time‑series JSON → clean numeric DataFrame."""
    df = pd.DataFrame.from_dict(raw, orient="index")

    # Rename columns: strip leading ``n. `` and Title‑case remainder → ``Open``/``Adj Close`` …
    rename_map = {
        col: re.sub(r"^\d+\.\s*", "", col).title() for col in df.columns
    }
    df.rename(columns=rename_map, inplace=True)

    # Convert everything **numeric** where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    # Datetime index + sort
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.sort_index(inplace=True)

    # Date filter
    return df.loc[START_DATE:END_DATE]


def _process_ticker(ticker: str) -> Optional[pd.Series]:
    fn, extra = _function_and_params()

    # ---- price request ----
    price_params = {
        "function": fn,
        "symbol": ticker,
        "outputsize": "full",
        "apikey": API_KEY_PRICE,
        **extra,
    }

    try:
        price_json = _call_av(price_http, price_params)
        ts_key = next(k for k in price_json if "Time Series" in k)
        price_df = _clean_price_df(price_json[ts_key])
        if price_df.empty:
            print(f"⏩  {ticker}: no price data in window")
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"⏩  {ticker}: price fetch failed ({exc})")
        return None

    # ---- fundamentals (optional) ----
    if DOWNLOAD_FUND:
        fund_params = {
            "function": "OVERVIEW",
            "symbol": ticker,
            "apikey": API_KEY_FUND,
        }
        try:
            fund_info = _call_av(fund_http, fund_params)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  {ticker}: fundamentals failed ({exc})")
            fund_info = {}
    else:
        fund_info = {}

    # ---- merge + save ----
    for k, v in fund_info.items():
        price_df[k] = v  # broadcast scalar across rows

    out_path = OUT_DIR / f"{ticker}.csv"
    price_df.to_csv(out_path, float_format="%.6f")

    return price_df["Close"].rename(ticker)  # for optional combined sheet

###############################################################################
#   Main                                                                      #
###############################################################################

def main() -> None:  # noqa: C901 – a bit long but readable
    if not API_KEY_PRICE:
        raise SystemExit("❌  Please set ALPHAVANTAGE_KEY env var or edit script.")

    print(
        f"Price RPM={PRICE_RPM} | Fund RPM={FUND_RPM} | "
        f"workers={WORKERS} | fundamentals={DOWNLOAD_FUND}"
    )

    combined = pd.DataFrame()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for series in tqdm(pool.map(_process_ticker, TICKERS), total=len(TICKERS), unit="ticker"):
            if SAVE_COMBINED and series is not None:
                combined = pd.concat([combined, series], axis=1)

    if SAVE_COMBINED and not combined.empty:
        combined.sort_index(inplace=True)
        out_combined = OUT_DIR / f"prices_combined_{GRANULARITY}_{START_DATE}_{END_DATE}.csv"
        combined.to_csv(out_combined, float_format="%.6f")
        print(f"✅ Combined price sheet saved → {out_combined.name}")

    print("Done – one CSV per ticker in", OUT_DIR.resolve())


if __name__ == "__main__":
    main()

