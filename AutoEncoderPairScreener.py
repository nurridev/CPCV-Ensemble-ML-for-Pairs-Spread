from datetime import datetime, timedelta
import pandas as pd
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import HDBSCAN
import numpy as np
from itertools import combinations
from tensorflow.keras.optimizers import AdamW
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
        Dense(10, activation='gelu', kernel_regularizer=l2(0.01)),
        Dropout(.2),
        Dense(10, activation='relu', kernel_regularizer=l2(0.01)), 
        Dropout(.1), 
        Dense(6,activation='gelu', kernel_regularizer=l2(0.01)),
        Dense(output_len,activation='gelu')
        ])
    model.compile(optimizer=AdamW(), loss='mse')
    return model

def stationary_screener(LOOKBACK, start, end, alpha, model):
    """
    Given start, end dates screens potential pairs
    Requires 3 * LOOKBACK worth of data 
    - Returns potential pairs where p-val <= alpha 
    """
    # Iterate through DIRECTORy
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]
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
        zscore_sublists = [rolling_zscore[max(i - LOOKBACK, 0):i] for i in range(len(rolling_zscore), 0, -LOOKBACK)]
        zscore_sublists.reverse() 
        x = y = np.array([zscore_sublists[-1]])
        x_val = y_val = np.array([zscore_sublists[-2]])
        early_stop = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True) # validation loss
        model.fit(x, y , epochs=1000, validation_data=(x_val, y_val), callbacks=early_stop) # Fit model with earling stopping callback
        # Encode newest data
        encoder = Sequential()
        for i in range(6):
            encoder.add(model.layers[i])
        stock_name = ''.join([letter for letter in file_name if letter.isupper()])
        stock_vect = (encoder.predict(x)).squeeze() 
        stock_vect.tolist()# Turns numpy into vanilla python lst 
        VECTOR_UNIVERSE[stock_name] = stock_vect.tolist()
# CLUSTERING
    cluster_model = HDBSCAN(min_cluster_size=6)
    print()
    data_labels = list(VECTOR_UNIVERSE.keys())
    vector_data = list(VECTOR_UNIVERSE.values())
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
    for assets in tqdm(combo_universe) :
        asset_data = {} 
        for name in assets:
            df = pd.read_csv(DIRECTORY + [f for f in files if name in f][0], index_col=0)
            df.index = pd.to_datetime(df.index) # Sets first column to datetime 
            target_date = pd.to_datetime(end_date)
            # Get the nearest index position
            pos = df.index.get_indexer([target_date], method='nearest')[0]

            # Get the 70 rows before that position
            df = df.iloc[max(0, pos - LOOKBACK) : pos] 
            asset_data[name] = df.Close.to_list() 
        
        p_val = multi_cointegration_test(asset_data)['p_value']
        final_p_vals[assets] = p_val 
    top_vals = [(item[0], item[1]) for item in final_p_vals.items() if item[1] <= alpha] # ((asset1,asset2), p_val)
    print(top_vals)
    idx = df_price.index[-1] 
    return top_vals, idx 
# FINAL LOOP 
train_amt = 20
lookback = 256
start_date = '2014-01-01'
end_date   = '2020-01-01'
obj_delimeter = '%Y-%m-%d'
model = build_model(lookback)

for i in range(train_amt): 
    end_date = datetime.strptime(end_date, obj_delimeter) 
    end_date = (end_date + timedelta(days=1)).strftime(obj_delimeter) # Add 1 time step 
    top_vals, dt = stationary_screener(lookback, start_date, end_date, .1, model)
    print(top_vals, dt)
    open('file.txt', 'a').write(str(top_vals))

