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
LOOKBACK  = 100 # Input into autoencoder
VECTOR_UNIVERSE = {}

def multi_cointegration_test(
    data_dict,
    det_order=0,
    k_ar_diff=1,
    significance=0.05
):
    """
    Test cointegration among N time series provided as lists, arrays, or pandas Series.

    - If N == 2: runs Phillips–Ouliaris (statsmodels.tsa.stattools.coint)
    - If N >= 3: runs Johansen test (statsmodels.tsa.vector_ar.vecm.coint_johansen)

              }
    """
    # 1) Convert inputs to pandas Series
    series_dict = {}
    for label, series in data_dict.items():
        if isinstance(series, pd.Series):
            series_dict[label] = series
        else:
            series_dict[label] = pd.Series(series)

    # 2) Align into a DataFrame and drop missing
    df = pd.concat(series_dict, axis=1).dropna()
    n = df.shape[1]

    # 3) Two-series case: Phillips-Ouliaris cointegration test
    if n == 2:
        y, x = df.iloc[:, 0], df.iloc[:, 1]
        t_stat, p_value, crit_vals = coint(y, x)
        return {
            'type': 'pairwise',
            't_stat': t_stat,
            'p_value': p_value,
            'critical_values': {'1%': crit_vals[0], '5%': crit_vals[1], '10%': crit_vals[2]},
            'is_cointegrated': p_value < significance
        }

    # 4) Multivariate case (>=3 series): Johansen test
    joh = coint_johansen(df, det_order, k_ar_diff)
    # Choose critical value column based on significance
    sig_to_col = {0.10: 0, 0.05: 1, 0.01: 2}
    cv_col = sig_to_col.get(significance, 1)

    # Determine cointegration rank using trace statistic
    rank = 0
    for i, trace_stat in enumerate(joh.lr1):
        if trace_stat > joh.cvt[i, cv_col]:
            rank += 1
        else:
            break

    return {
        'type': 'johansen',
        'nobs': df.shape[0],
        'variables': list(df.columns),
        'trace_stat': list(joh.lr1),
        'trace_cv': {
            '90%': list(joh.cvt[:, 0]),
            '95%': list(joh.cvt[:, 1]),
            '99%': list(joh.cvt[:, 2])
        },
        'maxeig_stat': list(joh.lr2),
        'maxeig_cv': {
            '90%': list(joh.cvm[:, 0]),
            '95%': list(joh.cvm[:, 1]),
            '99%': list(joh.cvm[:, 2])
        },
        'eigenvalues': list(joh.eig),
        'cointegration_rank': rank
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
        
        p_val = multi_cointegration_test(asset_data)['p_value']
        final_p_vals[assets] = p_val 
    top_vals = [(item[0], item[1]) for item in final_p_vals.items() if item[1] <= alpha] # ((asset1,asset2), p_val)
    print(top_vals)
    return top_vals 
# FINAL LOOP 

lookback = 256
start_date = '2014-01-01'
end_date   = '2020-01-01'
model = build_model(lookback)
if __name__ == "__main__":
    
    vector_universe, encoder = train_autoencoder(lookback, start_date, end_date, 64, model) # autoencodes vector Universe from directory
    clus = cluster_pvals(vector_universe, start_date, end_date, .1) # Cluster vector universe 

