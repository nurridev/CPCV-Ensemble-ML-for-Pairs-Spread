import requests
import pandas as pd
import time
import datetime
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy.linalg import eig
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm

# %%

api_key = 'SOYTY1WIARXAYK82'
start_date = '2010-1-1'
end_date   = '2023-1-1'


# %%
# List of tickers to download
tickers = [
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
tickers = ['AAPL', 'GGG', 'MA', 'V', 'TXG', 'DOCS', 'VOYA', 'XEL', 'TSLA']
# Function to download data for a single ticker and return the Close prices
def download_close_prices(ticker):
    """
    Downloads the Close price data for a given ticker from the Alpha Vantage API.
    Returns a DataFrame with the 'Close' price for that ticker.
    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}'
    
    try:
        response = requests.get(url)
        data = response.json()

        # Check if the response contains the expected data
        if 'Time Series (Daily)' not in data:
            print(f"Unexpected response structure for {ticker}. Retrying...")
            return None
        
        # Convert the time series data to a DataFrame
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')

        # Select the 'Close' column and rename it to the ticker symbol
        df['Close'] = df['4. close']  # Accessing the close prices
        df = df[['Close']]  # Keep only the Close column
        df.columns = [ticker]  # Rename the column to the ticker symbol
        
        return df

    except KeyError as e:
        print(f"KeyError for {ticker}: {e}.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"RequestException for {ticker}: {e}.")
        return None

# Main program to download and append Close data to a DataFrame
def download_data_for_tickers(ticker_list):
    """
    Downloads the Close prices for a list of tickers and appends them into a single DataFrame.
    """
    combined_data = pd.DataFrame()

    for ticker in tqdm(ticker_list, unit='ticker'):
        df = None
        while df is None:  # Retry if data is None (due to errors)
            df = download_close_prices(ticker)
            if df is None:
                print(f"Retrying for {ticker} after a delay...")
                time.sleep(61)  # Wait for a minute before retrying

        # Append the Close prices to the combined DataFrame
        if combined_data.empty:
            combined_data = df
        else:
            combined_data = pd.concat([combined_data, df], axis=1)

    return combined_data

# Download data for the specified tickers
if __name__ == '__main__':
    # Download and combine data
    ticker_data = download_data_for_tickers(tickers)
    
    # Print or save the result
    print(ticker_data)
    
#%%
# drop columns with nan between start date

ticker_data = ticker_data.sort_index()
ticker_data.index = pd.to_datetime(ticker_data.index)
ticker_data = ticker_data.loc[start_date:end_date]

ticker_data.dropna(axis=1, inplace=True)
ticker_data.to_csv('stock_data.csv')
