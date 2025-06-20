import threading
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy.linalg import eig
from statsmodels.tsa.tsatools import lagmat
import statsmodels.api as sm
from tqdm import tqdm
import pandas as pd
def johansen(series1, series2):
        data  = pd.concat([series1, series2], axis=1).dropna()
        
        # Perform Johansen test
        johansen_test = coint_johansen(data, det_order=0, k_ar_diff=1)
        return {
        'Cointegration Test Statistic': johansen_test.lr1,  # Trace statistic
        'p-values': johansen_test.cvt,  # Critical values (acts as an approximation of p-values)
        'strength': johansen_test.eig,
        'eigenvalues': johansen_test.evec
        }

def calculate_half_life(spread):
    # Calculate the lagged spread and the change in spread
    spread_lag = spread.shift(1)
    spread_delta = spread - spread_lag
    
    # Remove NaN values (due to shift)
    spread_lag = spread_lag[1:]
    spread_delta = spread_delta[1:]
    
    # Add a constant to the lagged spread for regression
    spread_lag = sm
    return half_life

ticker_data = pd.read_csv('stock_data.csv')
tickers = list(ticker_data.columns)
del tickers[0]

pair_list = list(combinations(tickers,2))
df_final = pd.DataFrame()
# iterate through every pair combo and perform test
for pair in tqdm(pair_list):
    try:
        series1 = ticker_data[pair[0]]
        series2 = ticker_data[pair[1]]
        
        # Calcualte difference 
        series1 = pd.to_numeric(series1, errors='coerce')
        series2 = pd.to_numeric(series2, errors='coerce')
        if series1.isnull().any()or series2.isnull().any():
            pass 
        johansen_results = johansen(series1,series2)
        max_strength = max(johansen_results['strength']) 
        idx_max =np.where(johansen_results['strength'] == max_strength)[0][0]



        df_johansen = pd.DataFrame({
                "Eigen1": johansen_results["eigenvalues"][0][idx_max],
                "Eigen2": johansen_results["eigenvalues"][1][idx_max],
                "strength": johansen_results['strength'][idx_max],
                "Pair1": pair[0],
                "Pair2": pair[1]
            },index=[0])
        
        df_final = pd.concat([df_johansen, df_final], ignore_index=True)
    except:
        pass
df_final.to_csv('pair_database.csv')















    
