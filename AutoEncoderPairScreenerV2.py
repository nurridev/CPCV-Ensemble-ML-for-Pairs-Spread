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
from statsmodels.tsa.stattools import adfuller, kpss
from tqdm import tqdm
DIRECTORY = '/home/kisero/ML-shyt-public/stock_data/'
VECTOR_UNIVERSE = {}

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def kalman_spread(y, x, q=1e-5, r=1e-3):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)

    beta_hat = np.zeros(n)
    P = 1.0

    for t in range(n):
        xt = x[t]

        # Predict
        beta_pred = beta_hat[t-1] if t > 0 else 0.0
        P = P + q

        # Update
        err = y[t] - beta_pred * xt
        S = xt * P * xt + r
        K = P * xt / S
        beta_new = beta_pred + K * err
        P = (1 - K * xt) * P

        beta_hat[t] = beta_new

    spread = y - beta_hat * x
    return beta_hat, spread

# ---------- Unit-root tests ----------
def run_adf(series):
    stat, p, _, _, _, _ = adfuller(series, autolag='AIC')
    return {"stat": stat, "p": p}

def run_kpss(series):
    stat, p, _, _ = kpss(series, regression='c', nlags='auto')
    return {"stat": stat, "p": p}

# ---------- Johansen with boolean significance ----------
def run_johansen(df, det_order=0, k_ar_diff=1):
    j = coint_johansen(df, det_order, k_ar_diff)

    # Criticals are shaped (r, 3) for 90/95/99
    levels = {"90": 0, "95": 1, "99": 2}
    trace_reject = {lv: (j.lr1 > j.cvt[:, idx]) for lv, idx in levels.items()}
    eigen_reject = {lv: (j.lr2 > j.cvm[:, idx]) for lv, idx in levels.items()}

    return {
        "trace_stat": j.lr1,
        "trace_crit": j.cvt,     # keep if you still want numbers
        "eigen_stat": j.lr2,
        "eigen_crit": j.cvm,
        "trace_reject": trace_reject,
        "eigen_reject": eigen_reject
    }

# ---------- Main wrapper ----------
def compute_metrics(price_dict, y_key=None, x_key=None):
    if y_key is None or x_key is None:
        keys = list(price_dict.keys())
        y_key, x_key = keys[0], keys[1]

    y = np.asarray(price_dict[y_key], dtype=float)
    x = np.asarray(price_dict[x_key], dtype=float)

    beta_hat, spread = kalman_spread(y, x)

    adf_res   = run_adf(spread)
    kpss_res  = run_kpss(spread)
    df_pair   = pd.DataFrame({y_key: y, x_key: x})
    joh_res   = run_johansen(df_pair)

    return {
        "kalman": {"beta": beta_hat, "spread": spread},
        "tests": {
            "adf": adf_res,
            "kpss": kpss_res,
            "johansen": joh_res
        }
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
        model.fit(np.array(x), np.array(y) , epochs=1, validation_data=(x_val, y_val), callbacks=early_stop) # Fit model with earling stopping callback
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
def cluster_pvals(vector_universe, start_date, end_date, alpha):
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
        # Check length
        length_ind = [len(item) for key, item in asset_data.items()] 
        aligned_length = all(x == length_ind[0] for x in length_ind) 
        if aligned_length:
            coint_info_test = compute_metrics(asset_data) 
            #final_p_vals[assets] = coint_info
            asset_data[name] = coint_info_test['tests']['adf']['p'], coint_info_test['tests']['kpss']['p'], coint_info_test['tests']['johansen']['trace_reject']['95'][0]
        else:
            print(f"Could not eval {asset_data.keys()}")
    top_vals = [(item[0], item[1]) for item in final_p_vals.items() if item[1][0] <= alpha and item[1][1] <= 1-alpha*2 and item[1][2]] # ((asset1,asset2), (adf-float, kpss-float, johansen-bool))
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
    clus = cluster_pvals(vector_universe, start_date, end_date, .1) # Cluster vector universe 

