#!/usr/bin/env python3
"""
🚀 AUTOENCODER PAIR SCREENER - COMPLETE DASHBOARD
===============================================

A comprehensive single-file interactive dashboard for discovering and analyzing
cointegrated stock pairs using deep learning autoencoders and statistical analysis.

This file contains everything needed to run the complete application:
- AutoEncoder pipeline (training, clustering, cointegration testing)
- Backend wrapper with caching and metrics calculation
- Interactive Dash dashboard with visualizations
- All statistical analysis functions (Kalman, Hurst, cointegration tests)

Features:
- Custom analysis date range selection
- Interactive pair selection and visualization
- Kalman-filtered spread plots with statistical bands
- 3D PCA projection of autoencoded outputs
- Cluster-based color coding
- Minimalist black & yellow theme
- Custom background image support
- Real-time metric display
- Intelligent result caching

Usage: python autoencoder_dashboard.py

Author: AI Assistant
Date: 2025
"""

import os
import sys
import pickle
import hashlib
import warnings
import random
import logging
import json
import base64
import io
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from itertools import combinations

# Suppress warnings for clean output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Data processing and ML imports
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN

# Deep learning imports
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.losses import Huber, MSE
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Statistical analysis imports
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats

# Dashboard imports
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Progress tracking
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# AUTOENCODER PIPELINE FUNCTIONS
# =============================================================================

def kalman_spread(y, x, q=1e-5, r=1e-3):
    """
    Calculate Kalman-filtered spread between two time series
    
    Args:
        y: Dependent variable time series
        x: Independent variable time series
        q: Process noise covariance
        r: Measurement noise covariance
        
    Returns:
        Tuple of (beta_coefficients, spread_series)
    """
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

def run_adf(series):
    """Run Augmented Dickey-Fuller test"""
    stat, p, _, _, _, _ = adfuller(series, autolag='AIC')
    return {"stat": stat, "p": p}

def run_kpss(series):
    """Run KPSS test"""
    stat, p, _, _ = kpss(series, regression='c', nlags='auto')
    return {"stat": stat, "p": p}

def run_johansen(df, det_order=0, k_ar_diff=1):
    """Run Johansen cointegration test with boolean significance"""
    j = coint_johansen(df, det_order, k_ar_diff)

    # Criticals are shaped (r, 3) for 90/95/99
    levels = {"90": 0, "95": 1, "99": 2}
    trace_reject = {lv: (j.lr1 > j.cvt[:, idx]) for lv, idx in levels.items()}
    eigen_reject = {lv: (j.lr2 > j.cvm[:, idx]) for lv, idx in levels.items()}

    return {
        "trace_stat": j.lr1,
        "trace_crit": j.cvt,
        "eigen_stat": j.lr2,
        "eigen_crit": j.cvm,
        "trace_reject": trace_reject,
        "eigen_reject": eigen_reject
    }

def compute_metrics(price_dict, y_key=None, x_key=None):
    """
    Compute comprehensive cointegration metrics for a pair
    
    Args:
        price_dict: Dictionary with stock names as keys, price lists as values
        y_key: Key for dependent variable (optional)
        x_key: Key for independent variable (optional)
        
    Returns:
        Dictionary with Kalman results and statistical tests
    """
    if y_key is None or x_key is None:
        keys = list(price_dict.keys())
        y_key, x_key = keys[0], keys[1]

    y = np.asarray(price_dict[y_key], dtype=float)
    x = np.asarray(price_dict[x_key], dtype=float)

    beta_hat, spread = kalman_spread(y, x)

    adf_res = run_adf(spread)
    kpss_res = run_kpss(spread)
    df_pair = pd.DataFrame({y_key: y, x_key: x})
    joh_res = run_johansen(df_pair)

    return {
        "kalman": {"beta": beta_hat, "spread": spread},
        "tests": {
            "adf": adf_res,
            "kpss": kpss_res,
            "johansen": joh_res
        }
    }

def build_model(output_len):
    """Build the autoencoder model"""
    model = Sequential([
        # First encoder layer
        Dense(224, activation='tanh', kernel_regularizer=l1_l2(l1=0.0001461896279370495, l2=0.0028016351587162596)),
        # Batch norm and dropout
        BatchNormalization(),
        Dropout(0.14881529393791154),
        # Second encoder layer
        Dense(int(224 * 0.75)),
        LeakyReLU(),
        # Latent layer
        Dense(12, activation='gelu'),
        # Decoder layers
        Dense(284, activation='gelu'),
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
    Create stride windows from a list
    
    Args:
        stride: Distance between each window
        lst: 1D list to apply stride to
        window: Size per sample
        
    Returns:
        2D array with windowed data
    """
    sample_amt = ((len(lst) - window) / stride) + 1
    wrap_backwards = False 
    result = []
    start = 0
    end = window  
    if isinstance(sample_amt, float):
        wrap_backwards = True
    for i in range(int(sample_amt)):
        result.append(lst[start:end]) 
        start += stride
        end += stride
    if wrap_backwards:
        result.append(lst[-window:])
    return np.array(result)

def train_autoencoder(LOOKBACK, start, end, stride, model, data_directory='./stock_data/Data/'):
    """
    Train autoencoder and return vector universe
    
    Args:
        LOOKBACK: Lookback window for training
        start: Start date string
        end: End date string  
        stride: Stride for window generation
        model: Keras model to train
        data_directory: Path to stock data
        
    Returns:
        Tuple of (vector_universe_dict, encoder_model)
    """
    files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
    vector_universe = {} 
    
    for file_name in files:
        try:
            df = pd.read_csv(os.path.join(data_directory, file_name), index_col=0)
            df = df[start:end]
            df_price = df['Close']
            
            if len(df_price) < LOOKBACK * 3:
                logger.warning(f'COULD NOT EVAL {file_name}: {len(df_price)} - insufficient data') 
                continue
                
            logger.info(f"Processing {file_name} with {len(df_price)} data points") 
            
            # Rolling z-score
            rolling_mean = df_price.rolling(window=LOOKBACK).mean()
            rolling_std = df_price.rolling(window=LOOKBACK).std()
            rolling_zscore = (df_price - rolling_mean) / rolling_std
            rolling_zscore = rolling_zscore.dropna(how='all').to_list()
            
            if len(rolling_zscore) < LOOKBACK + 6:  # Need enough data for validation
                logger.warning(f'COULD NOT EVAL {file_name}: insufficient zscore data after rolling calculation')
                continue
            
            x = y = stride_window(stride, rolling_zscore, LOOKBACK)
            
            if len(x) < 6:  # Need minimum samples for validation
                logger.warning(f'COULD NOT EVAL {file_name}: insufficient samples after stride windowing')
                continue
            
            # Random validation sets with bounds checking
            available_indices = list(range(1, len(x) - 2))
            if len(available_indices) < 3:
                logger.warning(f'COULD NOT EVAL {file_name}: insufficient indices for validation split')
                continue
                
            idx1, idx2, idx3 = random.sample(available_indices, 3) 
            x_val = y_val = np.array([x[idx1], x[idx2], x[idx3]])
            
            # Remove validation from training
            x = y = [x[sample_idx] for sample_idx in range(len(x)) if sample_idx not in [idx1, idx2, idx3]] 
            x, y = np.array(x), np.array(y) 
            
            if len(x) == 0:
                logger.warning(f'COULD NOT EVAL {file_name}: no training data after validation split')
                continue
            
            early_stop = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
            model.fit(np.array(x), np.array(y), epochs=1, validation_data=(x_val, y_val), callbacks=early_stop, verbose=0)
            
            # Create encoder
            encoder = Sequential()
            for i in range(6):  # First 6 layers up to latent space
                encoder.add(model.layers[i])
                
            stock_name = ''.join([letter for letter in file_name if letter.isupper()])
            
            # Get encoded vectors and ensure consistent shape
            stock_vect = encoder.predict(x, verbose=0)
            
            # Handle different output shapes - flatten to 1D if needed
            if len(stock_vect.shape) > 2:
                stock_vect = stock_vect.reshape(stock_vect.shape[0], -1)
            
            # Take mean across samples to get a single representative vector per stock
            if len(stock_vect.shape) == 2:
                stock_vect_mean = np.mean(stock_vect, axis=0)
            else:
                stock_vect_mean = stock_vect
            
            # Ensure we have a consistent vector length
            if hasattr(stock_vect_mean, 'shape') and len(stock_vect_mean.shape) > 0:
                vector_universe[stock_name] = stock_vect_mean.flatten().tolist()
                logger.info(f"Successfully processed {stock_name} - vector shape: {len(vector_universe[stock_name])}")
            else:
                logger.warning(f'COULD NOT EVAL {file_name}: invalid vector shape')
                continue
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            continue
            
    logger.info(f"Successfully processed {len(vector_universe)} stocks for vector universe")
    return vector_universe, encoder

# =============================================================================
# BACKEND WRAPPER CLASSES
# =============================================================================

class HurstExponentCalculator:
    """Calculate Hurst exponent for time series"""
    
    @staticmethod
    def calculate_hurst(ts: np.ndarray, max_lag: int = None) -> float:
        """
        Calculate Hurst exponent using R/S analysis
        
        Args:
            ts: Time series data
            max_lag: Maximum lag for calculation
            
        Returns:
            Hurst exponent (0.5 = random walk, >0.5 = trending, <0.5 = mean reverting)
        """
        try:
            ts = np.asarray(ts, dtype=float)
            n = len(ts)
            
            if max_lag is None:
                max_lag = n // 4
                
            if max_lag < 10:
                max_lag = min(10, n // 2)
                
            lags = np.arange(2, max_lag + 1)
            rs_values = []
            
            for lag in lags:
                # Divide series into non-overlapping subseries
                y = ts[:n - (n % lag)].reshape(-1, lag)
                
                # Calculate mean for each subseries
                means = np.mean(y, axis=1)
                
                # Calculate cumulative deviations
                deviations = np.cumsum(y - means[:, np.newaxis], axis=1)
                
                # Calculate range
                R = np.max(deviations, axis=1) - np.min(deviations, axis=1)
                
                # Calculate standard deviation
                S = np.std(y, axis=1, ddof=1)
                
                # Handle zero standard deviation
                S[S == 0] = 1e-10
                
                # Calculate R/S
                rs = R / S
                rs_values.append(np.mean(rs))
            
            # Linear regression on log-log plot
            log_lags = np.log(lags)
            log_rs = np.log(rs_values)
            
            # Remove any infinite or NaN values
            valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
            if np.sum(valid_mask) < 3:
                return 0.5
                
            log_lags = log_lags[valid_mask]
            log_rs = log_rs[valid_mask]
            
            # Calculate Hurst exponent as slope
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            
            # Clamp to reasonable range
            return max(0.0, min(1.0, hurst))
            
        except Exception as e:
            logger.warning(f"Error calculating Hurst exponent: {e}")
            return 0.5

class CacheManager:
    """Handle caching of analysis results"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters"""
        sorted_params = sorted(kwargs.items())
        params_str = json.dumps(sorted_params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def get_cached_result(self, **kwargs) -> Optional[Dict]:
        """Get cached result if exists"""
        cache_key = self._get_cache_key(**kwargs)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    result = json.load(f)
                logger.info(f"Retrieved cached result for key: {cache_key}")
                return result
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
                
        return None
    
    def cache_result(self, result: Dict, **kwargs) -> None:
        """Cache analysis result"""
        cache_key = self._get_cache_key(**kwargs)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            serializable_result = self._make_json_serializable(result)
            with open(cache_file, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            logger.info(f"Cached result for key: {cache_key}")
        except Exception as e:
            logger.warning(f"Error caching result: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

class AutoEncoderPairScreenerBackend:
    """Backend wrapper for AutoEncoderPairScreenerV2 pipeline"""
    
    def __init__(self, data_directory: str = None, cache_enabled: bool = True):
        """Initialize the backend"""
        self.data_directory = data_directory or './stock_data/Data/'
        self.cache_enabled = cache_enabled
        self.cache_manager = CacheManager() if cache_enabled else None
        self.hurst_calculator = HurstExponentCalculator()
        
        # Verify data directory exists
        if not os.path.exists(self.data_directory):
            logger.warning(f"Data directory not found: {self.data_directory}")
            
    def analyze_pairs(
        self,
        analysis_start_date: str,
        analysis_end_date: str,
        lookback: int = 256,
        stride: int = 64,
        alpha: float = 0.05,
        min_cluster_size: int = 6
    ) -> Dict[str, Any]:
        """Main analysis function to process historical stock data and find cointegrated pairs"""
        
        # Check cache first
        cache_params = {
            'analysis_start_date': analysis_start_date,
            'analysis_end_date': analysis_end_date,
            'lookback': lookback,
            'stride': stride,
            'alpha': alpha,
            'min_cluster_size': min_cluster_size,
            'data_directory': self.data_directory
        }
        
        if self.cache_enabled:
            cached_result = self.cache_manager.get_cached_result(**cache_params)
            if cached_result is not None:
                return cached_result
        
        logger.info(f"Starting analysis for period {analysis_start_date} to {analysis_end_date}")
        
        try:
            # Step 1: Train autoencoder and get vector universe
            model = build_model(lookback)
            vector_universe, encoder = train_autoencoder(
                lookback, analysis_start_date, analysis_end_date, stride, model, self.data_directory
            )
            
            if not vector_universe:
                return {
                    'status': 'error',
                    'message': 'No vectors extracted from autoencoder',
                    'pairs': [],
                    'metadata': {}
                }
            
            # Step 2: Perform clustering
            cluster_results = self._perform_clustering(vector_universe, min_cluster_size)
            
            # Step 3: Test cointegration for clustered pairs
            cointegration_results = self._test_cointegration(
                cluster_results['pairs'], analysis_start_date, analysis_end_date, alpha
            )
            
            # Step 4: Calculate additional metrics
            final_results = self._calculate_final_metrics(
                cointegration_results, analysis_start_date, analysis_end_date
            )
            
            # Step 5: Create PCA projection for visualization
            pca_data = self._create_pca_projection(vector_universe, cluster_results['cluster_labels'])
            
            # Compile final result
            result = {
                'status': 'success',
                'analysis_period': {
                    'start_date': analysis_start_date,
                    'end_date': analysis_end_date
                },
                'parameters': {
                    'lookback': lookback,
                    'stride': stride,
                    'alpha': alpha,
                    'min_cluster_size': min_cluster_size
                },
                'pairs': final_results,
                'pca_projection': pca_data,
                'metadata': {
                    'total_stocks': len(vector_universe),
                    'total_pairs_tested': len(cointegration_results),
                    'significant_pairs': len(final_results),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            # Cache the result
            if self.cache_enabled:
                self.cache_manager.cache_result(result, **cache_params)
            
            logger.info(f"Analysis completed. Found {len(final_results)} significant pairs.")
            return result
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'pairs': [],
                'metadata': {}
            }
    
    def _perform_clustering(self, vector_universe: Dict, min_cluster_size: int) -> Dict:
        """Perform HDBSCAN clustering on autoencoded vectors"""
        
        if not vector_universe:
            logger.warning("No vectors available for clustering")
            return {'pairs': [], 'cluster_labels': {}, 'cluster_groups': {}}
        
        try:
            cluster_model = HDBSCAN(min_cluster_size=min_cluster_size)
            data_labels = list(vector_universe.keys())
            vector_data = list(vector_universe.values())
            
            # Ensure all vectors are lists and have consistent dimensionality
            processed_vectors = []
            max_len = 0
            
            # First pass: determine max length and validate vectors
            valid_vectors = []
            valid_labels = []
            
            for i, (label, v) in enumerate(zip(data_labels, vector_data)):
                try:
                    if isinstance(v, (list, np.ndarray)):
                        v_flat = np.array(v).flatten()
                        if len(v_flat) > 0 and np.all(np.isfinite(v_flat)):
                            valid_vectors.append(v_flat.tolist())
                            valid_labels.append(label)
                            max_len = max(max_len, len(v_flat))
                        else:
                            logger.warning(f"Skipping {label}: invalid vector data")
                    else:
                        logger.warning(f"Skipping {label}: vector is not a list or array")
                except Exception as e:
                    logger.warning(f"Skipping {label}: error processing vector - {e}")
                    continue
            
            if len(valid_vectors) < min_cluster_size:
                logger.warning(f"Insufficient valid vectors ({len(valid_vectors)}) for clustering")
                return {'pairs': [], 'cluster_labels': {}, 'cluster_groups': {}}
            
            # Second pass: pad all vectors to same length
            for v in valid_vectors:
                if len(v) < max_len:
                    v_padded = v + [0.0] * (max_len - len(v))
                else:
                    v_padded = v[:max_len]
                processed_vectors.append(v_padded)
            
            # Convert to numpy array for clustering
            processed_vectors = np.array(processed_vectors)
            logger.info(f"Clustering {len(processed_vectors)} vectors of dimension {max_len}")
            
            # Perform clustering
            cluster_labels = cluster_model.fit_predict(processed_vectors)
            
            # Group stocks by cluster
            cluster_groups = {}
            for stock, cluster_id in zip(valid_labels, cluster_labels):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(stock)
            
            # Generate pairs from clusters (excluding noise cluster -1)
            pairs = []
            for cluster_id, stocks in cluster_groups.items():
                if cluster_id != -1 and len(stocks) >= 2:
                    cluster_pairs = list(combinations(stocks, 2))
                    pairs.extend(cluster_pairs)
                    logger.info(f"Cluster {cluster_id}: {len(stocks)} stocks, {len(cluster_pairs)} pairs")
            
            logger.info(f"Generated {len(pairs)} pairs from {len(cluster_groups)} clusters")
            
            return {
                'pairs': pairs,
                'cluster_labels': dict(zip(valid_labels, cluster_labels)),
                'cluster_groups': cluster_groups
            }
            
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            return {'pairs': [], 'cluster_labels': {}, 'cluster_groups': {}}
    
    def _test_cointegration(
        self, 
        pairs: List[Tuple[str, str]], 
        start_date: str, 
        end_date: str, 
        alpha: float
    ) -> List[Dict]:
        """Test cointegration for each pair"""
        
        results = []
        files = [f for f in os.listdir(self.data_directory) if f.endswith('.csv')]
        
        for stock1, stock2 in tqdm(pairs, desc="Testing cointegration"):
            try:
                # Find data files
                stock1_file = None
                stock2_file = None
                
                for f in files:
                    if stock1 in f:
                        stock1_file = f
                    if stock2 in f:
                        stock2_file = f
                
                if not stock1_file or not stock2_file:
                    continue
                
                # Read price data
                df1 = pd.read_csv(os.path.join(self.data_directory, stock1_file), index_col=0)
                df2 = pd.read_csv(os.path.join(self.data_directory, stock2_file), index_col=0)
                
                df1 = df1[start_date:end_date]
                df2 = df2[start_date:end_date]
                
                # Align dates
                common_dates = df1.index.intersection(df2.index)
                if len(common_dates) < 50:
                    continue
                
                df1_aligned = df1.loc[common_dates]
                df2_aligned = df2.loc[common_dates]
                
                # Prepare data for cointegration test
                asset_data = {
                    stock1: df1_aligned['Close'].tolist(),
                    stock2: df2_aligned['Close'].tolist()
                }
                
                # Perform cointegration tests
                coint_metrics = compute_metrics(asset_data, stock1, stock2)
                
                result = {
                    'pair': (stock1, stock2),
                    'data_points': len(common_dates),
                    'date_range': (common_dates[0], common_dates[-1]),
                    'metrics': coint_metrics,
                    'passes_tests': self._evaluate_cointegration(coint_metrics, alpha)
                }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing pair ({stock1}, {stock2}): {e}")
                continue
        
        return results
    
    def _evaluate_cointegration(self, metrics: Dict, alpha: float) -> bool:
        """Evaluate if pair passes cointegration tests"""
        try:
            adf_p = metrics['tests']['adf']['p']
            kpss_p = metrics['tests']['kpss']['p']
            johansen_reject = metrics['tests']['johansen']['trace_reject']['95'][0]
            
            adf_pass = adf_p <= alpha
            kpss_pass = kpss_p > alpha
            johansen_pass = johansen_reject
            
            return adf_pass and kpss_pass and johansen_pass
            
        except Exception:
            return False
    
    def _calculate_final_metrics(
        self, 
        cointegration_results: List[Dict], 
        start_date: str, 
        end_date: str
    ) -> List[Dict]:
        """Calculate final metrics for significant pairs"""
        
        final_pairs = []
        
        for result in cointegration_results:
            if not result['passes_tests']:
                continue
            
            try:
                stock1, stock2 = result['pair']
                metrics = result['metrics']
                
                # Extract Kalman filter results
                beta_hat = metrics['kalman']['beta']
                spread = metrics['kalman']['spread']
                
                # Calculate Hurst exponent
                hurst_exp = self.hurst_calculator.calculate_hurst(spread)
                
                # Calculate spread statistics
                spread_mean = np.mean(spread)
                spread_std = np.std(spread)
                
                # Get test p-values
                adf_p = metrics['tests']['adf']['p']
                kpss_p = metrics['tests']['kpss']['p']
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(adf_p, kpss_p, hurst_exp)
                
                pair_result = {
                    'pair': (stock1, stock2),
                    'kalman_spread': {
                        'series': spread,
                        'beta_coefficients': beta_hat,
                        'mean': spread_mean,
                        'std': spread_std,
                        'upper_band': spread_mean + spread_std,
                        'lower_band': spread_mean - spread_std
                    },
                    'cointegration_tests': {
                        'adf_p_value': adf_p,
                        'kpss_p_value': kpss_p,
                        'johansen_reject_95': metrics['tests']['johansen']['trace_reject']['95'][0]
                    },
                    'hurst_exponent': hurst_exp,
                    'confidence_score': confidence_score,
                    'data_points': result['data_points'],
                    'date_range': result['date_range']
                }
                
                final_pairs.append(pair_result)
                
            except Exception as e:
                logger.warning(f"Error calculating final metrics for pair {result['pair']}: {e}")
                continue
        
        # Sort by confidence score
        final_pairs.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return final_pairs
    
    def _calculate_confidence_score(self, adf_p: float, kpss_p: float, hurst_exp: float) -> float:
        """Calculate a confidence score for the pair"""
        try:
            # Lower ADF p-value is better
            adf_score = max(0, 1 - adf_p * 10)
            
            # Higher KPSS p-value is better
            kpss_score = min(1, kpss_p * 2)
            
            # Hurst exponent close to 0.5 indicates mean reversion
            hurst_score = 1 - abs(hurst_exp - 0.5) * 2
            
            # Weighted combination
            confidence = (adf_score * 0.4 + kpss_score * 0.4 + hurst_score * 0.2)
            
            return max(0, min(1, confidence))
            
        except Exception:
            return 0.5
    
    def _create_pca_projection(self, vector_universe: Dict, cluster_labels: Dict) -> Dict:
        """Create 3D PCA projection of autoencoded vectors"""
        
        try:
            # Prepare data
            stock_names = list(vector_universe.keys())
            vectors = list(vector_universe.values())
            
            # Ensure all vectors have the same length
            max_len = max(len(v) if isinstance(v, list) else 1 for v in vectors)
            processed_vectors = []
            
            for v in vectors:
                if isinstance(v, list):
                    if len(v) < max_len:
                        v_padded = v + [0] * (max_len - len(v))
                    else:
                        v_padded = v[:max_len]
                    processed_vectors.append(v_padded)
                else:
                    processed_vectors.append([v] + [0] * (max_len - 1))
            
            # Standardize vectors
            scaler = StandardScaler()
            vectors_scaled = scaler.fit_transform(processed_vectors)
            
            # Apply PCA
            pca = PCA(n_components=3)
            pca_coordinates = pca.fit_transform(vectors_scaled)
            
            # Create result
            pca_data = {
                'coordinates': pca_coordinates.tolist(),
                'stock_names': stock_names,
                'cluster_labels': [cluster_labels.get(stock, -1) for stock in stock_names],
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_explained_variance': float(np.sum(pca.explained_variance_ratio_))
            }
            
            return pca_data
            
        except Exception as e:
            logger.warning(f"Error creating PCA projection: {e}")
            return {
                'coordinates': [],
                'stock_names': [],
                'cluster_labels': [],
                'explained_variance_ratio': [],
                'total_explained_variance': 0.0
            }
    
    def get_pair_data_for_plotting(
        self, 
        pair: Tuple[str, str], 
        plot_start_date: str, 
        plot_end_date: str
    ) -> Dict[str, Any]:
        """Get detailed data for a specific pair for plotting"""
        try:
            stock1, stock2 = pair
            files = [f for f in os.listdir(self.data_directory) if f.endswith('.csv')]
            
            # Find data files
            stock1_file = None
            stock2_file = None
            
            for f in files:
                if stock1 in f:
                    stock1_file = f
                if stock2 in f:
                    stock2_file = f
            
            if not stock1_file or not stock2_file:
                raise ValueError(f"Data files not found for pair ({stock1}, {stock2})")
            
            # Load data
            df1 = pd.read_csv(os.path.join(self.data_directory, stock1_file), index_col=0)
            df2 = pd.read_csv(os.path.join(self.data_directory, stock2_file), index_col=0)
            
            # Filter by date range
            df1 = df1[plot_start_date:plot_end_date]
            df2 = df2[plot_start_date:plot_end_date]
            
            # Align dates
            common_dates = df1.index.intersection(df2.index)
            df1_aligned = df1.loc[common_dates]
            df2_aligned = df2.loc[common_dates]
            
            # Calculate Kalman spread
            y = df1_aligned['Close'].values
            x = df2_aligned['Close'].values
            beta_hat, spread = kalman_spread(y, x)
            
            # Calculate statistics
            spread_mean = np.mean(spread)
            spread_std = np.std(spread)
            
            return {
                'dates': common_dates.tolist(),
                'stock1_prices': y.tolist(),
                'stock2_prices': x.tolist(),
                'spread': spread.tolist(),
                'beta_coefficients': beta_hat.tolist(),
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'upper_band': spread_mean + spread_std,
                'lower_band': spread_mean - spread_std
            }
            
        except Exception as e:
            logger.error(f"Error getting plot data for pair {pair}: {e}")
            return {}

# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

# Color scheme and styling
COLORS = {
    'background': '#000000',
    'surface': '#1a1a1a',
    'primary': '#ffff00',     # Yellow
    'secondary': '#333333',
    'text': '#ffffff',
    'accent': '#ffcc00',
    'success': '#00ff00',
    'warning': '#ff8800',
    'error': '#ff0000'
}

# Custom CSS styles
CUSTOM_STYLE = {
    'backgroundColor': COLORS['background'],
    'color': COLORS['text'],
    'fontFamily': 'Arial, sans-serif',
    'minHeight': '100vh'
}

BUTTON_STYLE = {
    'backgroundColor': COLORS['primary'],
    'color': COLORS['background'],
    'border': 'none',
    'borderRadius': '5px',
    'padding': '10px 20px',
    'fontWeight': 'bold',
    'cursor': 'pointer'
}

# Initialize backend
backend = AutoEncoderPairScreenerBackend()

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.title = "🚀 AutoEncoder Pair Screener Dashboard"

# Global variables
analysis_results = {}

def create_header():
    """Create the dashboard header"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1(
                    "🚀 AutoEncoder Pair Screener Dashboard",
                    className="text-center",
                    style={'color': COLORS['primary'], 'fontWeight': 'bold', 'marginBottom': '20px'}
                ),
                html.P(
                    "Discover and analyze cointegrated stock pairs using advanced machine learning techniques",
                    className="text-center",
                    style={'color': COLORS['text'], 'fontSize': '18px'}
                )
            ])
        ])
    ], fluid=True, style={'padding': '20px'})

def create_control_panel():
    """Create the main control panel"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4("🎛️ Analysis Controls", style={'color': COLORS['primary'], 'margin': 0})
        ], style={'backgroundColor': COLORS['surface'], 'border': 'none'}),
        dbc.CardBody([
            dbc.Row([
                # Analysis Date Range
                dbc.Col([
                    html.Label("📅 Analysis Date Range", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.DatePickerRange(
                        id='analysis-date-range',
                        start_date=date(2017, 1, 1),
                        end_date=date(2019, 1, 1),
                        display_format='YYYY-MM-DD'
                    )
                ], width=6),
                
                # Analysis Parameters
                dbc.Col([
                    html.Label("⚙️ Parameters", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dbc.Row([
                        dbc.Col([
                            html.Small("Alpha", style={'color': COLORS['text']}),
                            dcc.Input(
                                id='alpha-input',
                                type='number',
                                value=0.1,
                                min=0.001,
                                max=0.2,
                                step=0.001,
                                style={
                                    'width': '100%',
                                    'backgroundColor': COLORS['surface'],
                                    'color': COLORS['text'],
                                    'border': f'1px solid {COLORS["primary"]}'
                                }
                            )
                        ], width=6),
                        dbc.Col([
                            html.Small("Min Cluster Size", style={'color': COLORS['text']}),
                            dcc.Input(
                                id='cluster-size-input',
                                type='number',
                                value=3,
                                min=2,
                                max=20,
                                step=1,
                                style={
                                    'width': '100%',
                                    'backgroundColor': COLORS['surface'],
                                    'color': COLORS['text'],
                                    'border': f'1px solid {COLORS["primary"]}'
                                }
                            )
                        ], width=6)
                    ])
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "🔍 Run Analysis",
                        id="run-analysis-btn",
                        color="warning",
                        size="lg",
                        style=BUTTON_STYLE
                    )
                ], width=3),
                dbc.Col([
                    html.Div(id="analysis-status", style={'color': COLORS['text'], 'padding': '10px'})
                ], width=9)
            ])
        ], style={'backgroundColor': COLORS['surface']})
    ], style={'backgroundColor': COLORS['surface'], 'border': f'1px solid {COLORS["primary"]}'})

def create_background_upload():
    """Create background image upload component"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("🖼️ Custom Background", style={'color': COLORS['primary'], 'margin': 0})
        ], style={'backgroundColor': COLORS['surface'], 'border': 'none'}),
        dbc.CardBody([
            dcc.Upload(
                id='upload-background',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Background Image', style={'color': COLORS['primary']})
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'borderColor': COLORS['primary'],
                    'textAlign': 'center',
                    'backgroundColor': COLORS['surface'],
                    'color': COLORS['text']
                }
            ),
            html.Div(id='background-status', style={'color': COLORS['text'], 'marginTop': '10px'})
        ], style={'backgroundColor': COLORS['surface']})
    ], style={'backgroundColor': COLORS['surface'], 'border': f'1px solid {COLORS["primary"]}'})

def create_pair_selector():
    """Create pair selection dropdown"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4("📊 Detected Pairs", style={'color': COLORS['primary'], 'margin': 0})
        ], style={'backgroundColor': COLORS['surface'], 'border': 'none'}),
        dbc.CardBody([
            dcc.Dropdown(
                id='pair-dropdown',
                placeholder="Select a cointegrated pair...",
                style={
                    'backgroundColor': COLORS['surface'],
                    'color': COLORS['background']
                }
            )
        ], style={'backgroundColor': COLORS['surface']})
    ], style={'backgroundColor': COLORS['surface'], 'border': f'1px solid {COLORS["primary"]}'})

def create_plotting_controls():
    """Create plotting date range controls"""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("📈 Plotting Controls", style={'color': COLORS['primary'], 'margin': 0})
        ], style={'backgroundColor': COLORS['surface'], 'border': 'none'}),
        dbc.CardBody([
            html.Label("📅 Plotting Date Range", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
            dcc.DatePickerRange(
                id='plotting-date-range',
                start_date=date(2017, 1, 1),
                end_date=date(2019, 1, 1),
                display_format='YYYY-MM-DD'
            ),
            html.Br(),
            dbc.Switch(
                id="show-pca-filter",
                label="Show only selected pairs in PCA",
                value=False,
                style={'color': COLORS['text'], 'marginTop': '10px'}
            )
        ], style={'backgroundColor': COLORS['surface']})
    ], style={'backgroundColor': COLORS['surface'], 'border': f'1px solid {COLORS["primary"]}'})

def create_spread_plot():
    """Create spread plot container"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4("📊 Kalman-Filtered Spread Analysis", style={'color': COLORS['primary'], 'margin': 0})
        ], style={'backgroundColor': COLORS['surface'], 'border': 'none'}),
        dbc.CardBody([
            dcc.Graph(
                id='spread-plot',
                style={'height': '500px'},
                config={'displayModeBar': False}
            )
        ], style={'backgroundColor': COLORS['surface']})
    ], style={'backgroundColor': COLORS['surface'], 'border': f'1px solid {COLORS["primary"]}'})

def create_pca_plot():
    """Create PCA 3D plot container"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4("🎯 3D PCA Projection", style={'color': COLORS['primary'], 'margin': 0})
        ], style={'backgroundColor': COLORS['surface'], 'border': 'none'}),
        dbc.CardBody([
            dcc.Graph(
                id='pca-plot',
                style={'height': '600px'},
                config={'displayModeBar': False}
            )
        ], style={'backgroundColor': COLORS['surface']})
    ], style={'backgroundColor': COLORS['surface'], 'border': f'1px solid {COLORS["primary"]}'})

def create_metrics_display():
    """Create metrics display component"""
    return html.Div(id='metrics-display')

# Main layout
app.layout = html.Div([
    # Background div for custom background
    html.Div(id='background-div', style={
        'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 
        'height': '100%', 'zIndex': -1, 'backgroundColor': COLORS['background']
    }),
    
    # Main content
    html.Div([
        create_header(),
        
        dbc.Container([
            # Control Panel Row
            dbc.Row([
                dbc.Col([create_control_panel()], width=8),
                dbc.Col([create_background_upload()], width=4)
            ], className="mb-4"),
            
            # Pair Selection and Controls Row
            dbc.Row([
                dbc.Col([create_pair_selector()], width=6),
                dbc.Col([create_plotting_controls()], width=6)
            ], className="mb-4"),
            
            # Metrics Display Row
            dbc.Row([
                dbc.Col([create_metrics_display()], width=12)
            ], className="mb-4"),
            
            # Visualization Row
            dbc.Row([
                dbc.Col([create_spread_plot()], width=7),
                dbc.Col([create_pca_plot()], width=5)
            ])
        ], fluid=True)
    ], style=CUSTOM_STYLE)
], style={'backgroundColor': COLORS['background']})

# =============================================================================
# DASHBOARD CALLBACKS
# =============================================================================

@app.callback(
    [Output('analysis-status', 'children'),
     Output('pair-dropdown', 'options'),
     Output('pair-dropdown', 'value')],
    [Input('run-analysis-btn', 'n_clicks')],
    [State('analysis-date-range', 'start_date'),
     State('analysis-date-range', 'end_date'),
     State('alpha-input', 'value'),
     State('cluster-size-input', 'value')]
)
def run_analysis(n_clicks, start_date, end_date, alpha, cluster_size):
    """Run the pair analysis"""
    if n_clicks is None:
        return "Ready to analyze...", [], None
    
    global analysis_results
    
    try:
        # Run analysis
        results = backend.analyze_pairs(
            analysis_start_date=str(start_date),
            analysis_end_date=str(end_date),
            alpha=alpha,
            min_cluster_size=cluster_size
        )
        
        analysis_results = results
        
        if results['status'] == 'success':
            if results['pairs']:
                # Create enhanced dropdown options with comprehensive statistics
                options = []
                for i, pair_data in enumerate(results['pairs']):
                    stock1, stock2 = pair_data['pair']
                    confidence = pair_data['confidence_score']
                    hurst = pair_data['hurst_exponent']
                    adf_p = pair_data['cointegration_tests']['adf_p_value']
                    kpss_p = pair_data['cointegration_tests']['kpss_p_value']
                    johansen_reject = pair_data['cointegration_tests']['johansen_reject_95']
                    data_points = pair_data['data_points']
                    
                    # Create comprehensive label with statistics
                    label = (f"{stock1}/{stock2} | "
                            f"Conf:{confidence:.3f} | "
                            f"Hurst:{hurst:.3f} | "
                            f"ADF:{adf_p:.4f} | "
                            f"KPSS:{kpss_p:.3f} | "
                            f"Johan:{'✓' if johansen_reject else '✗'} | "
                            f"N:{data_points}")
                    
                    options.append({'label': label, 'value': i})
                
                status_msg = html.Div([
                    html.Span("✅ Analysis complete! ", style={'color': COLORS['success']}),
                    html.Span(f"Found {len(results['pairs'])} cointegrated pairs from {results['metadata'].get('total_pairs_tested', 0)} tested pairs", style={'color': COLORS['text']})
                ])
                
                return status_msg, options, 0
            else:
                status_msg = html.Div([
                    html.Span("⚠️ No cointegrated pairs found. ", style={'color': COLORS['warning']}),
                    html.Span(f"Tested {results['metadata'].get('total_pairs_tested', 0)} pairs. Try adjusting parameters.", style={'color': COLORS['text']})
                ])
                return status_msg, [], None
        else:
            error_msg = html.Div([
                html.Span("❌ Analysis failed: ", style={'color': COLORS['error']}),
                html.Span(results.get('message', 'Unknown error'), style={'color': COLORS['text']})
            ])
            return error_msg, [], None
            
    except Exception as e:
        error_msg = html.Div([
            html.Span("❌ Error: ", style={'color': COLORS['error']}),
            html.Span(str(e), style={'color': COLORS['text']})
        ])
        logger.error(f"Error in run_analysis callback: {e}")
        return error_msg, [], None

@app.callback(
    Output('metrics-display', 'children'),
    [Input('pair-dropdown', 'value')]
)
def update_metrics_display(selected_pair_idx):
    """Update the metrics display for the selected pair"""
    if selected_pair_idx is None or not analysis_results.get('pairs'):
        return html.Div()
    
    pair_data = analysis_results['pairs'][selected_pair_idx]
    stock1, stock2 = pair_data['pair']
    
    metrics_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 Pair", style={'color': COLORS['primary']}),
                    html.H4(f"{stock1} / {stock2}", style={'color': COLORS['text']})
                ])
            ], style={'backgroundColor': COLORS['surface'], 'border': f'1px solid {COLORS["primary"]}'})
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🎯 P-Value (ADF)", style={'color': COLORS['primary']}),
                    html.H4(f"{pair_data['cointegration_tests']['adf_p_value']:.4f}", 
                            style={'color': COLORS['success'] if pair_data['cointegration_tests']['adf_p_value'] < 0.05 else COLORS['warning']})
                ])
            ], style={'backgroundColor': COLORS['surface'], 'border': f'1px solid {COLORS["primary"]}'})
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📈 Hurst Exponent", style={'color': COLORS['primary']}),
                    html.H4(f"{pair_data['hurst_exponent']:.4f}", style={'color': COLORS['text']})
                ])
            ], style={'backgroundColor': COLORS['surface'], 'border': f'1px solid {COLORS["primary"]}'})
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🔍 KPSS P-Value", style={'color': COLORS['primary']}),
                    html.H4(f"{pair_data['cointegration_tests']['kpss_p_value']:.4f}", style={'color': COLORS['text']})
                ])
            ], style={'backgroundColor': COLORS['surface'], 'border': f'1px solid {COLORS["primary"]}'})
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("⭐ Confidence", style={'color': COLORS['primary']}),
                    html.H4(f"{pair_data['confidence_score']:.4f}", style={'color': COLORS['accent']})
                ])
            ], style={'backgroundColor': COLORS['surface'], 'border': f'1px solid {COLORS["primary"]}'})
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📅 Data Points", style={'color': COLORS['primary']}),
                    html.H4(f"{pair_data['data_points']}", style={'color': COLORS['text']})
                ])
            ], style={'backgroundColor': COLORS['surface'], 'border': f'1px solid {COLORS["primary"]}'})
        ], width=2)
    ])
    
    return metrics_cards

@app.callback(
    Output('spread-plot', 'figure'),
    [Input('pair-dropdown', 'value'),
     Input('plotting-date-range', 'start_date'),
     Input('plotting-date-range', 'end_date')]
)
def update_spread_plot(selected_pair_idx, plot_start, plot_end):
    """Update the spread plot"""
    if selected_pair_idx is None or not analysis_results.get('pairs'):
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['surface'],
            font=dict(color=COLORS['text']),
            title="Select a pair to view spread analysis"
        )
        return fig
    
    pair_data = analysis_results['pairs'][selected_pair_idx]
    stock1, stock2 = pair_data['pair']
    
    try:
        # Get plotting data
        plot_data = backend.get_pair_data_for_plotting(
            pair=(stock1, stock2),
            plot_start_date=str(plot_start),
            plot_end_date=str(plot_end)
        )
        
        if not plot_data:
            fig = go.Figure()
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['surface'],
                font=dict(color=COLORS['text']),
                title="No data available for selected date range"
            )
            return fig
        
        dates = pd.to_datetime(plot_data['dates'])
        spread = plot_data['spread']
        mean_val = plot_data['spread_mean']
        upper_band = plot_data['upper_band']
        lower_band = plot_data['lower_band']
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=[f"{stock1} / {stock2} Spread Analysis", "Spread Distribution"],
            vertical_spacing=0.1
        )
        
        # Main spread plot
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=spread,
                mode='lines',
                name='Kalman Spread',
                line=dict(color=COLORS['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Statistical bands
        fig.add_hline(y=mean_val, line=dict(color=COLORS['text'], dash='dash', width=1), row=1, col=1)
        fig.add_hline(y=upper_band, line=dict(color=COLORS['warning'], dash='dot', width=1), row=1, col=1)
        fig.add_hline(y=lower_band, line=dict(color=COLORS['warning'], dash='dot', width=1), row=1, col=1)
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=spread,
                nbinsx=50,
                name='Distribution',
                marker=dict(color=COLORS['accent'], opacity=0.7)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['surface'],
            font=dict(color=COLORS['text']),
            height=500,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['surface'],
            font=dict(color=COLORS['text']),
            title=f"Error loading data: {str(e)}"
        )
        return fig

@app.callback(
    Output('pca-plot', 'figure'),
    [Input('pair-dropdown', 'value'),
     Input('show-pca-filter', 'value')]
)
def update_pca_plot(selected_pair_idx, show_only_selected):
    """Update the 3D PCA plot"""
    if not analysis_results.get('pca_projection'):
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['surface'],
            font=dict(color=COLORS['text']),
            title="Run analysis to view PCA projection"
        )
        return fig
    
    pca_data = analysis_results['pca_projection']
    coordinates = np.array(pca_data['coordinates'])
    stock_names = pca_data['stock_names']
    cluster_labels = pca_data['cluster_labels']
    
    if len(coordinates) == 0:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['surface'],
            font=dict(color=COLORS['text']),
            title="No PCA data available"
        )
        return fig
    
    # Define colors for clusters
    unique_clusters = list(set(cluster_labels))
    cluster_colors = {}
    color_palette = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', 
                     '#ff8000', '#8000ff', '#80ff00', '#0080ff']
    
    for i, cluster in enumerate(unique_clusters):
        if cluster == -1:
            cluster_colors[cluster] = '#666666'
        else:
            cluster_colors[cluster] = color_palette[i % len(color_palette)]
    
    # Filter data if requested
    if show_only_selected and selected_pair_idx is not None and analysis_results.get('pairs'):
        selected_pair = analysis_results['pairs'][selected_pair_idx]['pair']
        filter_stocks = list(selected_pair)
        filter_indices = [i for i, stock in enumerate(stock_names) if stock in filter_stocks]
        
        if filter_indices:
            coordinates = coordinates[filter_indices]
            stock_names = [stock_names[i] for i in filter_indices]
            cluster_labels = [cluster_labels[i] for i in filter_indices]
    
    fig = go.Figure()
    
    # Add traces for each cluster
    for cluster in unique_clusters:
        cluster_mask = np.array(cluster_labels) == cluster
        if not np.any(cluster_mask):
            continue
            
        cluster_coords = coordinates[cluster_mask]
        cluster_stocks = [stock_names[i] for i in range(len(stock_names)) if cluster_mask[i]]
        
        cluster_name = f'Cluster {cluster}' if cluster != -1 else 'Noise'
        
        fig.add_trace(
            go.Scatter3d(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                z=cluster_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=cluster_colors[cluster],
                    opacity=0.8,
                    line=dict(width=1, color=COLORS['text'])
                ),
                text=cluster_stocks,
                hovertemplate='<b>%{text}</b><br>' +
                             'PC1: %{x:.3f}<br>' +
                             'PC2: %{y:.3f}<br>' +
                             'PC3: %{z:.3f}<br>' +
                             f'<b>{cluster_name}</b><extra></extra>',
                name=cluster_name
            )
        )
    
    # Highlight selected pair
    if selected_pair_idx is not None and analysis_results.get('pairs'):
        selected_pair = analysis_results['pairs'][selected_pair_idx]['pair']
        for stock in selected_pair:
            if stock in stock_names:
                stock_idx = stock_names.index(stock)
                fig.add_trace(
                    go.Scatter3d(
                        x=[coordinates[stock_idx, 0]],
                        y=[coordinates[stock_idx, 1]],
                        z=[coordinates[stock_idx, 2]],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color=COLORS['primary'],
                            symbol='diamond',
                            line=dict(width=3, color=COLORS['background'])
                        ),
                        text=[stock],
                        name=f'Selected: {stock}',
                        showlegend=False
                    )
                )
    
    explained_var = pca_data.get('explained_variance_ratio', [0, 0, 0])
    total_var = pca_data.get('total_explained_variance', 0)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['surface'],
        font=dict(color=COLORS['text']),
        title=f"3D PCA Projection (Total Variance Explained: {total_var:.1%})",
        scene=dict(
            xaxis_title=f"PC1 ({explained_var[0]:.1%})" if len(explained_var) > 0 else "PC1",
            yaxis_title=f"PC2 ({explained_var[1]:.1%})" if len(explained_var) > 1 else "PC2",
            zaxis_title=f"PC3 ({explained_var[2]:.1%})" if len(explained_var) > 2 else "PC3",
            bgcolor=COLORS['surface'],
            xaxis=dict(gridcolor=COLORS['secondary']),
            yaxis=dict(gridcolor=COLORS['secondary']),
            zaxis=dict(gridcolor=COLORS['secondary'])
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

@app.callback(
    Output('background-div', 'style'),
    [Input('upload-background', 'contents')]
)
def update_background(contents):
    """Update background image"""
    base_style = {
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'width': '100%',
        'height': '100%',
        'zIndex': -1,
        'backgroundColor': COLORS['background']
    }
    
    if contents is not None:
        try:
            content_type, content_string = contents.split(',')
            base64_url = f"data:image/png;base64,{content_string}"
            
            base_style.update({
                'backgroundImage': f'url({base64_url})',
                'backgroundSize': 'cover',
                'backgroundPosition': 'center',
                'backgroundRepeat': 'no-repeat',
                'opacity': '0.1'
            })
        except Exception:
            pass
    
    return base_style

@app.callback(
    Output('background-status', 'children'),
    [Input('upload-background', 'contents')]
)
def update_background_status(contents):
    """Update background upload status"""
    if contents is not None:
        return html.Span("✅ Background image uploaded", style={'color': COLORS['success']})
    return html.Span("No background image", style={'color': COLORS['text']})

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def print_startup_banner():
    """Print startup information"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                🚀 AUTOENCODER PAIR SCREENER 🚀                ║
║                     Complete Dashboard                       ║
╠═══════════════════════════════════════════════════════════════╣
║  🎯 Discover cointegrated stock pairs using ML autoencoders  ║
║  📊 Advanced statistical analysis and visualization          ║
║  ⚡ Real-time interactive dashboard with caching            ║
║  🔬 Complete pipeline in a single file                      ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("🚀 Starting AutoEncoder Pair Screener Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    print("🎯 Features enabled:")
    print("   • Custom analysis date ranges")
    print("   • Interactive pair selection")
    print("   • Kalman-filtered spread visualization")
    print("   • 3D PCA projection with clustering")
    print("   • Custom background image support")
    print("   • Real-time caching")
    print("   • Complete pipeline integration")
    print("\n💡 TIP: Start with default date range (2017-2019) for optimal performance")
    print("💡 TIP: Use Alpha=0.1 and Min Cluster Size=3 for more pairs")
    print("\n🌐 Dashboard ready! Visit: http://localhost:8050")
    print("🛑 Press Ctrl+C to stop the dashboard\n")

if __name__ == '__main__':
    print_startup_banner()
    app.run_server(debug=True, host='0.0.0.0', port=8050) 