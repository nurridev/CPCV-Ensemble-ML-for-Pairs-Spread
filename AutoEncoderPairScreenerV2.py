import pickle
import random 
from datetime import datetime, timedelta
import pandas as pd
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.losses import Huber, MSE
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import HDBSCAN
import numpy as np
from itertools import combinations
from tensorflow.keras.optimizers import Adam
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm
DIRECTORY = '/home/kisero/ML-shyt-public/stock_data/'
VECTOR_UNIVERSE = {}

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def _hurst_exponent(ts, lags_range=range(2, 100)):
    """Compute the Hurst exponent of a time series using a simple R/S method.

    Parameters
    ----------
    ts : array‑like
        1‑D sequence of numeric observations.
    lags_range : iterable of int, optional
        Sequence of lags over which to compute the rescaled range.  The default
        range(2, 100) gives a quick yet reasonably stable estimate.  Increase
        the upper bound for a more precise but slower calculation.

    Returns
    -------
    float
        Estimated Hurst exponent H where:
        * H < 0.5  → anti‑persistent / mean‑reverting
        * H ≈ 0.5  → uncorrelated / random walk
        * H > 0.5  → persistent / trending
    """
    ts = np.asarray(ts, dtype=float)
    lags = np.asarray(list(lags_range))
    # Compute tau for each lag: sqrt of the variance of lagged differences
    tau = [np.sqrt(np.var(ts[lag:] - ts[:-lag])) for lag in lags]
    # Linear fit of log(lag) vs log(tau) → slope = H/2
    slope, _ = np.polyfit(np.log(lags), np.log(tau), 1)
    return 2.0 * slope


def _hurst_exponent(ts, lags_range=range(2, 100)):
    '''Compute the (global) Hurst exponent of a time‑series using a simple
    R/S‑style log‑log regression on the standard deviation of lagged differences.
    Returns
    -------
    float
        Estimated Hurst exponent H \in (0,1). 0.5 ≈ random walk; <0.5 mean‑reverting;
        >0.5 trending/persistent.'''  # noqa: E501
    ts = np.asarray(ts)
    if ts.ndim != 1:
        raise ValueError('`ts` must be 1‑dimensional')
    # Standard deviation of differenced series at multiple lags
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags_range]
    # Linear regression in log‑space: log(tau) = c + H * log(lag)
    poly = np.polyfit(np.log(lags_range), np.log(tau), 1)
    hurst = 2.0 * poly[0]
    return float(hurst)


def _kalman_spread(y: pd.Series, x: pd.Series, q=.0005, r=.0005) -> pd.Series:
    '''Estimate a dynamic hedge ratio β_t and intercept α_t between two series
    via a simple 2‑state Kalman filter, then return the resulting spread

        spread_t = y_t − β_t * x_t − α_t

    Parameters
    ----------
    y, x : pd.Series (must be aligned)
    q : float, optional
        Process (state‑transition) noise variance. Larger → more time‑varying β/α.
    r : float, optional
        Observation noise variance.

    Returns
    -------
    pd.Series
        The inferred spread series (same index as inputs).'''  # noqa: E501
    if not y.index.equals(x.index):
        raise ValueError('Series indices must match for Kalman filter regression.')

    n = len(y)
    # State vector [β, α]
    state = np.zeros(2)
    P = np.eye(2)  # State covariance
    Q = q * np.eye(2)
    R = r
    beta, alpha = np.zeros(n), np.zeros(n)

    for t in range(n):
        # ---- Prediction ----
        P = P + Q  # F = I, so x̂ = x, P̂ = P + Q

        # ---- Update ----
        H = np.array([x.iloc[t], 1.0]).reshape(1, 2)  # Observation matrix
        y_pred = (H @ state.reshape(-1, 1))[0, 0]
        S = (H @ P @ H.T)[0, 0] + R
        K = (P @ H.T).flatten() / S  # Kalman gain (2‑vector)

        state = state + K * (y.iloc[t] - y_pred)
        P = (np.eye(2) - np.outer(K, H)) @ P

        beta[t], alpha[t] = state  # Save for inspection if needed

    spread = y - beta * x - alpha
    return pd.Series(spread, index=y.index, name='spread')



def multi_cointegration_test(
    data_dict,
    det_order: int = 0,
    k_ar_diff: int = 1,
    significance: float = 0.05,
):
    '''
    Test for cointegration among ≥2 time‑series AND compute a *single* Hurst
    exponent on the resulting spread.

    Workflow
    --------
    • N = 2  →  Phillips–Ouliaris coint test + Kalman filter regression to
               extract a dynamic spread (y|x)  →  Hurst(spread).
    • N ≥ 3 →  Johansen trace test. The first cointegration vector (column 0)
               is used to form a static spread  →  Hurst(spread).

    Returns
    -------
    dict
        Always contains keys:
          • 'type'                – 'pairwise' | 'johansen'
          • 'hurst_exponent'      – float, H(spread)
          • 'spread'              – pd.Series of the derived spread
        …plus the usual statistics from the chosen cointegration test.
    '''  # noqa: E501

    # 1) Normalise input → aligned DataFrame
    series_dict = {lbl: (ser if isinstance(ser, pd.Series) else pd.Series(ser))
                   for lbl, ser in data_dict.items()}
    df = pd.concat(series_dict, axis=1).dropna()
    n = df.shape[1]

    if n < 2:
        raise ValueError('Need at least two series for cointegration analysis.')

    # -------------------------- Pairwise (N = 2) --------------------------- #
    if n == 2:
        y, x = df.iloc[:, 0], df.iloc[:, 1]
        # Kalman‑filter spread & Hurst
        spread = _kalman_spread(y, x)
        hurst = _hurst_exponent(spread.dropna())

        # Phillips–Ouliaris test
        t_stat, p_value, crit_vals = coint(y, x)
        return {
            'type': 'pairwise',
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'critical_values': {'1%': crit_vals[0], '5%': crit_vals[1], '10%': crit_vals[2]},
            'is_cointegrated': bool(p_value < significance),
            'spread_method': 'kalman',
            'spread': spread,
            'hurst_exponent': hurst,
        }

    # ------------------------ Multivariate (N ≥ 3) ------------------------- #
    joh = coint_johansen(df, det_order, k_ar_diff)

    # Map α to critical‑value column
    sig_to_col = {0.10: 0, 0.05: 1, 0.01: 2}
    cv_col = sig_to_col.get(significance, 1)

    # Trace‑statistic rank selection
    rank = 0
    for stat, cv in zip(joh.lr1, joh.cvt[:, cv_col]):
        if stat > cv:
            rank += 1
        else:
            break

    # Use first cointegration vector to derive *static* spread
    coint_vec = joh.evec[:, 0]
    spread_values = df.values @ coint_vec
    spread = pd.Series(spread_values, index=df.index, name='spread')
    hurst = _hurst_exponent(spread)

    return {
        'type': 'johansen',
        'nobs': int(df.shape[0]),
        'variables': list(df.columns),
        'trace_stat': list(map(float, joh.lr1)),
        'trace_cv': {
            '90%': list(map(float, joh.cvt[:, 0])),
            '95%': list(map(float, joh.cvt[:, 1])),
            '99%': list(map(float, joh.cvt[:, 2])),
        },
        'maxeig_stat': list(map(float, joh.lr2)),
        'maxeig_cv': {
            '90%': list(map(float, joh.cvm[:, 0])),
            '95%': list(map(float, joh.cvm[:, 1])),
            '99%': list(map(float, joh.cvm[:, 2])),
        },
        'eigenvalues': list(map(float, joh.eig)),
        'cointegration_rank': int(rank),
        'spread_method': 'johansen_first_vec',
        'spread_vector': list(map(float, coint_vec)),
        'spread': spread,
        'hurst_exponent': hurst,
    }
# AUTOENCODING
def build_model(output_len):
    model = Sequential([
        # First encoder layer
        Dense(224, activation='tanh', kernel_regularizer=l1_l2(l1=0.0001461896279370495, l2=0.0028016351587162596)),
        # Optional batch norm and dropout
        BatchNormalization(),
        Dropout(0.14881529393791154),
        # Second encoder layer (decreased units, using leaky_relu)
        Dense(int(224 * 0.75)),  # decrease strategy
        LeakyReLU(),
        # Latent layer
        Dense(12, activation='gelu'),
        # Decoder layer 1
        Dense(284, activation='gelu'),
        # Decoder layer 0 batch norm and dropout
        BatchNormalization(),
        Dropout(0.2123738038749523),
        # Output layer
        Dense(output_len, activation='gelu')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.018274508859816022),
        loss=Huber()
    )
    return model
def stride_window(stride, lst, window):
    """
    stride: distance between each window
    window: size per sample 
    lst   : 1D list to apply stride to

    returns: 2D lst with (len(lst) - window) + 1) , window)
    """
    sample_amt = ((len(lst) - window) / stride) + 1
    wrap_backwards = False 
    result = []
    start  = 0
    end    = window  
    if isinstance(sample_amt,float): # Not even overlap
        wrap_backwards = True
    for i in range(int(sample_amt)):
        result.append(lst[start:end]) 
        start += stride
        end   += stride
    if wrap_backwards:
        result.append(lst[-window:])
    return np.array(result)
def train_autoencoder(LOOKBACK, start, end, stride, model):
    """
    Given start, end dates trains based on LOOKBACK rolling window
    Requires 3 * LOOKBACK worth of data 
    - Returns dict: stock_name: vectors 
    """
    # Iterate through DIRECTORy
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]
    vector_universe = {} 
    for file_name in files:
        df = pd.read_csv(DIRECTORY + file_name, index_col=0) # Assumes date is first index
        df = df[start:end] # only include start-end 
        df_price = df['Close']
        if len(df_price) < LOOKBACK*3: # 3 cuz zscore_lookback validation train
            print(f'COULD NOT EVAL {file_name}: {len(df_price)}') 
            continue
        print(f"Loaded {file_name} {len(df_price)}") 
        # Rolling z-score
        rolling_mean = df_price.rolling(window=LOOKBACK).mean()
        rolling_std = df_price.rolling(window=LOOKBACK).std()
        rolling_zscore = (df_price - rolling_mean) / rolling_std
        rolling_zscore = rolling_zscore.dropna(how='all').to_list()
        x = y = stride_window(stride, rolling_zscore, LOOKBACK )
        # Random validation sets
        idx1, idx2, idx3 = random.sample(range(1, len(x) - 2), 3) 
        x_val = y_val = np.array([x[idx1], x[idx2], x[idx3]])
        # Subtract validation sets from train
        x = y = [x[sample_idx] for sample_idx in range(len(x)) if sample_idx not in [idx1,idx2,idx3]] 
        x, y = np.array(x), np.array(y) 
        early_stop = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True) # earlystop function
        model.fit(np.array(x), np.array(y) , epochs=10000, validation_data=(x_val, y_val), callbacks=early_stop) # Fit model with earling stopping callback
        # Encode newest data
        encoder = Sequential()
        for i in range(6): #range(n) where n is layer of latent space
            encoder.add(model.layers[i])
        stock_name = ''.join([letter for letter in file_name if letter.isupper()])
        #return model  
        
        stock_vect = encoder.predict(x).squeeze() # Squeezes and predicts the most re 
        stock_vect.tolist()# Turns numpy into vanilla python lst 
        vector_universe[stock_name] = stock_vect.tolist()
    return vector_universe, encoder        
# CLUSTERING
def cluster_pvals(vector_universe, start_date, end_date, alpha, hurst):
    """
    -Given vector_universe ->  {'Stock_name':[samples, latent_vector], 'Stock_name':...}
    -Returns


    """ 
    cluster_model = HDBSCAN(min_cluster_size=6)
    data_labels = list(vector_universe.keys())
    vector_data = list(vector_universe.values())
    flatten_to_2d = lambda data: [[i for sub in ([el] if not isinstance(el, list) else el) for i in (sub if not isinstance(sub, list) else [x for x in sub])] if not isinstance(item, list) else [i for sub in item for i in (sub if not isinstance(sub, list) else [x for x in sub])] for item in data]
    pad = lambda lst: [x + [0] * (max(map(len, lst)) - len(x)) for x in lst]
    vector_data = pad(flatten_to_2d(vector_data))
    cluster_set = cluster_model.fit_predict(vector_data).tolist() # Fit and predict cluster algo 
     
     
    VECTOR_CLUSTER = dict(zip(data_labels, cluster_set))
    sorted_clusters = {}
    combo_universe = [] # every combo in each cluster
    # Sort Clusters
    for key, value in VECTOR_CLUSTER.items():
        if value not in sorted_clusters.keys():
            sorted_clusters[value] = [key]
        else:
            sorted_clusters[value].append(key)
    for key, lst in sorted_clusters.items():
        if len(lst) < 2:
            pass
        else:
            combos = list(combinations(lst, 2))
            for item in combos:
                combo_universe.append(item)
# Iterate through all pairs
    final_p_vals = {}
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]   
    for assets in tqdm(combo_universe) :
        asset_data = {} 
        for name in assets:
            df = pd.read_csv(DIRECTORY + [f for f in files if name in f][0], index_col=0)
            df = df[start_date:end_date] 
            df.index = pd.to_datetime(df.index) # Sets first column to datetime 
            asset_data[name] = df.Close.to_list() 
        coint_info_test = multi_cointegration_test(asset_data) 
         
        final_p_vals[assets] = (coint_info_test['p_value'], coint_info_test['hurst_exponent'])
    top_vals = [(item[0], item[1][0], item[1][1]) for item in final_p_vals.items() if item[1][0] <= alpha and item[1][1] <= hurst] # ((asset1,asset2), (p_val, hurst)), (...)
    print(top_vals)
    return top_vals 
# FINAL LOOP 

lookback = 256
start_date = '2012-01-01'
end_date   = '2018-01-01'
model = build_model(lookback)
if __name__ == "__main__":
    
    vector_universe, encoder = train_autoencoder(lookback, start_date, end_date, 64, model) # autoencodes vector Universe from directory
    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoder, f) 
    clus = cluster_pvals(vector_universe, start_date, end_date, .1, .5) # Cluster vector universe 

