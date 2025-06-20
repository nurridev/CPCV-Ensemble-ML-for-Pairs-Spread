import joblib
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.dummy import DummyClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from netcal.metrics import ACE
from netcal.scaling import LogisticCalibration
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibrationDisplay
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import os
from sklearn.metrics import precision_score, recall_score
import optuna as optuna
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from itertools import combinations
import keras_tuner as kt
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
import glob

"""
- Takes random combinations of n .csv files from /labeled_data
- Performs like a batched CPCV 
- Combines multiple models using a meta learner

- Takes CSV files where stock_name represents series and label represents -1 0 1 labels





"""
DIRECTORY = '/home/kisero/ML_shyt/labeled_data/'
def preProXGB(x,y):
    
    # StandardScaler data
    x = StandardScaler().fit_transform(x)
    y = [i if i != -1 else 2 for i in y] 

    return x, y

def threshold(probs):
    """
    - Takes 2d arr (samples, classes)
    - returns 2d arr (samples, [pred_class])
    """
    THRESH = .3
    thresh_probs = []
    argmax = np.argmax(probs,axis=1)
    for max_idx, sample in zip(argmax,probs):
        max_class = sample[max_idx]
        if max_class >= THRESH:
            thresh_probs.append(max_idx)
        else:
            thresh_probs.append(0)


    return thresh_probs
def calculate_indicators(df, lookback) -> pd.DataFrame:
    """
    -Takes pandas Dataframe
    - First column is price series
    - Returns x, y numpy arr with indicators and label (respectivly)

    """

    # Price series
    x = df[df.columns[0]]
     
    # do a lil cleaning (get rid of index column) 
    if 'index' in df.columns:
        df = df.drop('index',axis=1)
    # Moving Averages
    df['sma'] = x.rolling(lookback).mean()
    df['ema'] = x.ewm(span=lookback, adjust=False).mean()
    weights = np.arange(1, lookback + 1)
    df['wma'] = x.rolling(lookback).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    df['hma'] = x.rolling(lookback).apply(
        lambda prices: np.sqrt(lookback) * (2 * prices[-lookback//2:].mean() - prices.mean()), raw=False
    )
    df['tema'] = 3*x.ewm(span=lookback, adjust=False).mean() - 3*x.ewm(span=lookback, adjust=False).mean().ewm(span=lookback, adjust=False).mean() + x.ewm(span=lookback, adjust=False).mean().ewm(span=lookback, adjust=False).mean().ewm(span=lookback, adjust=False).mean()
    
    # MACD and Signal
    ema12 = x.ewm(span=12, adjust=False).mean()
    ema26 = x.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = x.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(lookback).mean()
    avg_loss = loss.rolling(lookback).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # CCI (Close-only adaptation)
    typical_price = x
    ma = typical_price.rolling(lookback).mean()
    md = (typical_price - ma).abs().rolling(lookback).mean()
    df['cci'] = (typical_price - ma) / (0.015 * md)
    
    # Momentum
    df['roc'] = x.pct_change(lookback)
    df['momentum'] = x - x.shift(lookback)
    
    # Z-score
    mean = x.rolling(lookback).mean()
    std = x.rolling(lookback).std()
    df['zscore'] = (x - mean) / std
    
    # Percent Rank
    df['percent_rank'] = x.rolling(lookback).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1])
    
    # Daily return
    df['daily_return'] = x.pct_change()
    
    # Normalized price
    df['normalized'] = (x - x.rolling(lookback).min()) / (x.rolling(lookback).max() - x.rolling(lookback).min())
    
    # Price Oscillator
    short_ma = x.rolling(lookback).mean()
    long_ma = x.rolling(lookback * 2).mean()
    df['price_oscillator'] = (short_ma - long_ma) / long_ma
    
    # Slope (linear regression on Close)
    def slope(series):
        y = series.values
        x_vals = np.arange(len(y))
        if len(y) < 2:
            return np.nan
        A = np.vstack([x_vals, np.ones(len(x_vals))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return m
    df['slope'] = x.rolling(lookback).apply(slope, raw=False)
    
    # Lag
    df['lag'] = x.shift(lookback)
    
    # Rolling std (volatility)
    df['volatility'] = x.rolling(lookback).std()  
    
    # Envelope (SMA bands)
    ma = x.rolling(lookback).mean()
    envelope_pct = 0.02
    df['envelope_upper'] = ma * (1 + envelope_pct)
    df['envelope_lower'] = ma * (1 - envelope_pct)
    
    # Coppock Curve
    roc11 = x.pct_change(11)
    roc14 = x.pct_change(14)
    df['coppock'] = (roc11 + roc14).rolling(10).sum()
    
    # Diff
    df['diff'] = x.diff()
    
    # Williams %R (approx using close only)
    rolling_max = x.rolling(lookback).max()
    rolling_min = x.rolling(lookback).min()
    df['williams_r'] = -100 * ((rolling_max - x) / (rolling_max - rolling_min + 1e-9))
    # Remove Nans
    df.dropna(how='any',inplace=True) 
    df.set_index(df.columns[0],inplace=True) 
    label = df.label.to_numpy() 
    # Drop label
    df.drop('label',axis=1, inplace=True)
    features = df.to_numpy()
    return features, label

#%%
def data_split(data, n):
    size = len(data) // n
    remainder = len(data) % n
    splits = []
    start = 0
    
    for i in range(n):
        extra = 1 if i < remainder else 0
        end = start + size + extra
        splits.append([start, end])
        start = end
    return splits

def generate_combo_splits(n_groups, n_train_splits):
    all_idx = list(range(n_groups))
    combos = list(combinations(all_idx, n_train_splits))
    return combos


def split_combos(n_groups, n_train_splits):
    # Create list of tuples ([test_indexes], [train_combos])
    test_combos = generate_combo_splits(n_groups, n_train_splits)
    length_arr = list(range(n_groups))
    cpcv_splits = []
    
    for test_combo in test_combos:
        test_arr = [int(i) for i in test_combo]
        train_arr = [idx for idx in length_arr if idx not in test_combo]
        cpcv_splits.append((test_arr, train_arr))
    
    return cpcv_splits

def intervals_overlap(a_start, a_end, b_start, b_end):
    
    return not (a_end <= b_start or b_end <= a_start)
#%%       
def purge_embargo(fold_ranges, combos, purge, embargo):
    
    for (test_folds, train_fold) in combos:
        test_indices = []
        train_indices = []
        
        # Extend train_indices
        for trf in train_fold:
            tr_start, tr_end = fold_ranges[trf]
            train_indices.extend(range(tr_start, tr_end))
        for tf in test_folds:
            t_start, t_end = fold_ranges[tf]
            # Apply purging
            t_start += purge
            t_end   -= purge + embargo
            # Extend ranges from start-end
            test_indices.extend(range(t_start, t_end))
            
            # Check all combinations for overlap
            for trf in train_fold:
                tr_start, tr_end = fold_ranges[trf]
                if intervals_overlap(tr_start, tr_end, t_start, t_end):
                    print(f"Overlap in test fold {trf} and train fold {tf}")
        yield test_indices, train_indices


# df and split num
 


            
def eval_model(model_build, x, y, purge, embargo, lookback, fold_ranges,combos):    
    scores = []    
    for test_indices, train_indices in purge_embargo(fold_ranges, combos, purge, embargo):
         
        # Index training data given purged and embargoed indexes (convert to numpy for rolling_window func)
        x_train = x[train_indices]
        x_test  = x[test_indices]
        y_train = y[train_indices]
        y_test  = y[test_indices]
        
        # Split Train for calibration 
        """
        - x_train used for training base models 
        - x_cal used for metamodel
        - x_val used for calibration of full ensembles
        """
        # Take calibration and validation from test set3
        x_test, x_cal, y_test, y_cal = train_test_split(x_test,y_test, test_size= .2,shuffle=True, stratify=y_test)
        x_test, x_val, y_test, y_val = train_test_split(x_test,y_test, test_size= .2, shuffle=True,stratify=y_test)
        # Preprocess train and test 
        x_cal, y_cal = preProXGB(x_cal, y_cal)
        x_val, y_val = preProXGB(x_cal, y_cal) 
        x_train, y_train = preProXGB(x_train, y_train)
        x_test , y_test  = preProXGB(x_test,y_test) 
        # init and build model from updated hyperparams
        models, meta_model, platt= model_build()
        model_pred_cal= []
        model_pred_val = []
        model_pred_test = []
        for model in models:
            model.fit(x_train, y_train)           
             
            # Squeeze for Catboost output  
            model_pred_cal.append(model.predict(x_cal).squeeze())
            model_pred_val.append(model.predict(x_val).squeeze())
            model_pred_test.append(model.predict(x_test).squeeze())
        # Make model prediction into [[model1,model2,...], [model1,model2,...]] (transposition)
        model_pred_cal = [list(pair) for pair in zip(*model_pred_cal)]
        model_pred_val = [list(pair) for pair in zip(*model_pred_val)]
        model_pred_test = [list(pair) for pair in zip(*model_pred_test)]
        # Train metalearner CALIBRATION
        meta_model.fit(model_pred_cal, y_cal)

        model_pred_val = meta_model.predict_proba(model_pred_val) # Converts into uncalibrated probabilities from metalearner 
        # Fit calibration model 
        platt.fit(model_pred_val, np.array(y_val))  
        model_pred_test= platt.transform(np.array(meta_model.predict_proba(model_pred_test))) 
        ace = ACE(bins=8)
        confidence = ace.measure(model_pred_test, np.array(y_test))
        model_pred_test = threshold(model_pred_test) # Threshold  
        report_summary = classification_report(y_test, model_pred_test, output_dict=True, zero_division=0)
        # Print Calibration 
        # Final score is the average between class 1 and 2 precision
        score = (report_summary['1.0']['precision'] + report_summary['2.0']['precision']) / 2

        print(classification_report(y_test,model_pred_test,zero_division=0)) 
        print(score) 
        scores.append(score)   
    return np.mean(scores)
def objective(trial): 

    def model_build():
        
    
        #Returns list of estimator models, and META-MODELj
        
        # --- XGBoost ---
        param_xgb = {
            'n_estimators':    trial.suggest_int('xgb_n_estimators', 300, 500),
            'max_depth':       trial.suggest_int('xgb_max_depth', 3, 12),
            'learning_rate':   trial.suggest_float('xgb_lr', 1e-3, 0.3, log=True),
            'subsample':       trial.suggest_float('xgb_subsample', 0.5, 1.0),
            'colsample_bytree':trial.suggest_float('xgb_colsample', 0.5, 1.0),
            'gamma':           trial.suggest_float('xgb_gamma', 0.0, 10.0),
            'reg_alpha':       trial.suggest_float('xgb_alpha', 1e-8, 10.0, log=True),
            'reg_lambda':      trial.suggest_float('xgb_lambda', 1e-8, 10.0, log=True),
            'use_label_encoder': True,
            'eval_metric':     'logloss',
        }

        # --- Random Forest ---
        param_rf = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 500),
            'max_depth':    trial.suggest_int('rf_max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('rf_min_samples_leaf', 1, 20),
        }

        # --- Extra Trees ---
        param_et = {
            'n_estimators': trial.suggest_int('et_n_estimators', 50, 500),
            'max_depth':    trial.suggest_int('et_max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('et_min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('et_min_samples_leaf', 1, 20),
        }

        # --- HistGradientBoosting ---
        param_hgb = {
            'max_iter':     trial.suggest_int('hgb_max_iter', 50, 300),
            'max_leaf_nodes': trial.suggest_int('hgb_max_leaf_nodes', 15, 100),
            'learning_rate':  trial.suggest_float('hgb_lr', 1e-3, 0.3, log=True),
            'l2_regularization': trial.suggest_float('hgb_l2', 0.0, 5.0),
            'max_depth':      trial.suggest_int('hgb_max_depth', 3, 20),
        }

        # --- LightGBM ---
        param_lgb = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 500),
            'num_leaves':   trial.suggest_int('lgb_num_leaves', 16, 256),
            'learning_rate': trial.suggest_float('lgb_lr', 1e-3, 0.3, log=True),
            'subsample':     trial.suggest_float('lgb_subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('lgb_colsample', 0.5, 1.0),
            'reg_alpha':     trial.suggest_float('lgb_alpha', 1e-8, 10.0, log=True),
            'reg_lambda':    trial.suggest_float('lgb_lambda', 1e-8, 10.0, log=True),
            'verbose' :  -1
            }

        # --- CatBoost ---
        param_cat = {
            'iterations':       trial.suggest_int('cat_iter', 50, 500),
            'depth':            trial.suggest_int('cat_depth', 3, 12),
            'learning_rate':    trial.suggest_float('cat_lr', 1e-3, 0.3, log=True),
            'l2_leaf_reg':      trial.suggest_float('cat_l2', 1e-8, 10.0, log=True),
            'verbose':          False,
        }

        # --- SVM --a
        param_svc = {
            'C':        trial.suggest_float('svc_C', 1e-3, 10.0, log=True),
            'kernel':   trial.suggest_categorical('svc_kernel', ['rbf', 'poly', 'sigmoid']),
            'gamma':    trial.suggest_categorical('svc_gamma', ['scale', 'auto']),
            'probability': True,
        }

        

        # --- K-Nearest Neighbors ---
        param_knn = {
            'n_neighbors': trial.suggest_int('knn_n', 3, 50),
            'weights':     trial.suggest_categorical('knn_weights', ['uniform', 'distance']),
            'p':           trial.suggest_int('knn_p', 1, 2),
        }

        # --- MLP Classifier ---
        param_mlp = {
            'hidden_layer_sizes': trial.suggest_categorical('mlp_layers', [(50,), (100,), (50,50)]),
            'alpha':              trial.suggest_float('mlp_alpha', 1e-5, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('mlp_lr', 1e-4, 1e-1, log=True),
            'max_iter':           500,
        }
        param_logistic = {
        'C':               trial.suggest_float('log_C', 1e-4, 1e2, log=True),
        'tol':             trial.suggest_float('log_tol', 1e-6, 1e-2, log=True),
        'fit_intercept':   trial.suggest_categorical('log_fit_intercept', [True, False]),
        'class_weight':    trial.suggest_categorical('log_class_weight', [None, 'balanced']),
        'max_iter':        trial.suggest_int('log_max_iter', 100, 1000),
        # only used when penalty='elasticnet'
        'l1_ratio':        trial.suggest_float('log_l1_ratio', 0.0, 1.0),
        } 
        param_logistic_cal = {
        'penalty':      trial.suggest_categorical('calib_penalty', ['l2', 'l1', 'elasticnet', 'none']),
        'C':            trial.suggest_float('calib_C', 1e-4, 1e2, log=True),
        'solver':       trial.suggest_categorical('calib_solver', ['lbfgs', 'liblinear', 'saga']),
        'tol':          trial.suggest_float('calib_tol', 1e-6, 1e-2, log=True),
        'max_iter':     trial.suggest_int('calib_max_iter', 100, 1000),
        'l1_ratio':     trial.suggest_float('calib_l1_ratio', 0.0, 1.0),
        }
     
        models = [
        # original ensemble members
        xgb.XGBClassifier(**param_xgb),
        RandomForestClassifier(**param_rf),
        ExtraTreesClassifier(**param_et),
        HistGradientBoostingClassifier(**param_hgb),
        LGBMClassifier(**param_lgb),
        CatBoostClassifier(**param_cat),
        SVC(**param_svc),
        KNeighborsClassifier(**param_knn),
        MLPClassifier(**param_mlp),
        GaussianNB(),

        # added for diversity (no new Optuna params)
        DecisionTreeClassifier(
            max_depth=param_rf['max_depth'],
            min_samples_split=param_rf['min_samples_split'],
            min_samples_leaf=param_rf['min_samples_leaf'],
            random_state=0
        ),
        LogisticRegression(**param_logistic),
        BernoulliNB(),
        LinearDiscriminantAnalysis(),
        BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=10,
            random_state=0
        ),

        # extra non-linear models
        AdaBoostClassifier(),                                    # boosting of stumps
        QuadraticDiscriminantAnalysis(),                         # class‐conditional Gaussians
        GaussianProcessClassifier(),                             # kernelized Bayes
        ]
        
        #models = [RandomForestClassifier(**param_rf), xgb.XGBClassifier(**param_xgb)]        
        meta_model = LogisticRegression(**param_logistic)
        #meta_model = MLPClassifier(hidden_layer_sizes=())
        platt      = LogisticCalibration(**param_logistic_cal) 
        # store model hyperparams 
        trial.set_user_attr("model", models)
        trial.set_user_attr("meta-model", meta_model)
        trial.set_user_attr("calibration", platt)
        return models, meta_model, platt
    # Using random chosen files find average CPCV score
    layer_2_scores = []
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))] 
    random.seed(69) 
    chosen_files = random.choices(files, k=2) 
    for file_name in chosen_files:
        df = pd.read_csv(DIRECTORY + file_name) 
        df = df.dropna(how='any') 
        if 'index' in df.columns:
            df = df.drop('index',axis=1)
        # First column is price series
        data = pd.DataFrame({'price':df[df.columns[0]].to_numpy(), 'label': df.label.to_numpy()})
        # Calculate indicators
        x, y = calculate_indicators(data, 150)  
        
        # Calculate split ranges and amount of combos  
        fold_ranges =  data_split(data, 3) 
        combos      = split_combos(3, 1) # train-sets, test-sets 
        layer_2_score = eval_model(model_build, x, y, purge=15, embargo=15, lookback=30, fold_ranges=fold_ranges, combos=combos)
        layer_2_scores.append(layer_2_score)
    print(np.mean(layer_2_scores)) 
    return np.mean(layer_2_scores)
 


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=300)
print("Best parameters:", study.best_params)
print("Best value (objective):", study.best_value)

#model evaluation


best_trial = study.best_trial
models = best_trial.user_attrs["model"]
meta = best_trial.user_attrs["meta-model"]
calibrate  = best_trial.user_attrs["calibration"]
joblib.dump(models, 'best_models')
joblib.dump(meta, 'meta_model')
joblib.dump(calibrate, 'calibration_layer')
