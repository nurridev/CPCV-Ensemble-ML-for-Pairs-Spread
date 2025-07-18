import pickle
import random 
from datetime import datetime, timedelta
import pandas as pd
import os
import numpy as np
from itertools import combinations
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm
from sklearn.cluster import HDBSCAN
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

# Configuration
DIRECTORY = 'stock_data/Data/'
LOOKBACK = 256  # Must match training lookback
ENCODER_PATH = "encoder.pkl"

def get_user_preferences():
    """
    Get user preferences for number of stocks per group
    """
    print("\n" + "="*60)
    print("🚀 AUTOENCODER PAIRS/TRIPLETS SCREENER CONFIGURATION")
    print("="*60)
    
    while True:
        try:
            n_stocks = int(input("\n📊 How many stocks per group do you want to analyze?\n"
                                "   2 = Pairs trading\n"
                                "   3 = Triplets trading\n"
                                "   4+ = N-tuple trading\n"
                                "Enter number (2-10): "))
            
            if 2 <= n_stocks <= 10:
                break
            else:
                print("❌ Please enter a number between 2 and 10")
        except ValueError:
            print("❌ Please enter a valid number")
    
    print(f"\n✅ Selected: {n_stocks}-tuple trading")
    if n_stocks == 2:
        print("📈 You'll be analyzing stock pairs")
    elif n_stocks == 3:
        print("📈 You'll be analyzing stock triplets")
    else:
        print(f"📈 You'll be analyzing {n_stocks}-stock combinations")
    
    return n_stocks

def multi_cointegration_test(
    data_dict,
    det_order=0,
    k_ar_diff=1,
    significance=0.05
):
    """
    Test cointegration among N time series provided as lists, arrays, or pandas Series.
    """
    # Convert inputs to pandas Series
    series_dict = {}
    for label, series in data_dict.items():
        if isinstance(series, pd.Series):
            series_dict[label] = series
        else:
            series_dict[label] = pd.Series(series)

    # Align into a DataFrame and drop missing
    df = pd.concat(series_dict, axis=1).dropna()
    n = df.shape[1]

    # Two-series case: Phillips-Ouliaris cointegration test
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

    # Multivariate case (>=3 series): Johansen test
    joh = coint_johansen(df, det_order, k_ar_diff)
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

def process_single_cluster(cluster_id, stock_list, start_date, end_date, n_stocks, alpha, files):
    """
    Process a single cluster to find cointegrated N-tuples
    This function is designed to be run in parallel
    """
    cluster_results = []
    
    if len(stock_list) < n_stocks:
        return cluster_results
    
    # Generate combinations within this cluster only
    cluster_combos = list(combinations(stock_list, n_stocks))
    
    print(f"🔍 Cluster {cluster_id}: Testing {len(cluster_combos)} {n_stocks}-tuples")
    
    # Test cointegration for combinations in this cluster
    for assets in cluster_combos:
        try:
            asset_data = {}
            for name in assets:
                matching_files = [f for f in files if name in f]
                if not matching_files:
                    continue
                    
                df = pd.read_csv(DIRECTORY + matching_files[0], index_col=0)
                df = df[start_date:end_date]
                df.index = pd.to_datetime(df.index)
                asset_data[name] = df.Close.to_list()
            
            if len(asset_data) == n_stocks:
                p_val = multi_cointegration_test(asset_data)['p_value']
                if p_val <= alpha:  # Only keep results that pass alpha threshold
                    cluster_results.append((assets, p_val))
                    
        except Exception as e:
            # Silently continue on errors to avoid cluttering output
            continue
    
    print(f"✅ Cluster {cluster_id}: Found {len(cluster_results)} cointegrated {n_stocks}-tuples")
    return cluster_results

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
    if isinstance(sample_amt, float):
        wrap_backwards = True
    for i in range(int(sample_amt)):
        result.append(lst[start:end]) 
        start += stride
        end   += stride
    if wrap_backwards:
        result.append(lst[-window:])
    return np.array(result)

def load_encoder():
    """Load the trained encoder from pickle file"""
    try:
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        print(f"✅ Successfully loaded encoder from {ENCODER_PATH}")
        return encoder
    except FileNotFoundError:
        print(f"❌ Error: {ENCODER_PATH} not found. Please run training first.")
        return None
    except Exception as e:
        print(f"❌ Error loading encoder: {e}")
        return None

def screen_stocks(encoder, start_date, end_date, n_stocks=2, stride=64, alpha=0.05):
    """
    Screen stocks using the trained encoder and same preprocessing pipeline
    """
    print(f"🔍 Screening {n_stocks}-tuples from {start_date} to {end_date}")
    print(f"📊 Using alpha = {alpha} for cointegration test")
    
    # Get vector universe using trained encoder
    vector_universe = get_vector_universe(encoder, start_date, end_date, stride)
    
    if not vector_universe:
        print("❌ No valid stock data found!")
        return []
    
    print(f"📈 Found {len(vector_universe)} valid stocks for screening")
    
    # Find cointegrated tuples using clustering (now multithreaded)
    cointegrated_tuples = cluster_pvals_multithreaded(vector_universe, start_date, end_date, n_stocks, alpha)
    
    if not cointegrated_tuples:
        print(f"❌ No cointegrated {n_stocks}-tuples found with alpha = {alpha}")
        return []
    
    print(f"✅ Found {len(cointegrated_tuples)} cointegrated {n_stocks}-tuples")
    
    # Sort by p-value and take top 10
    sorted_tuples = sorted(cointegrated_tuples, key=lambda x: x[1])[:10]
    
    return sorted_tuples

def get_vector_universe(encoder, start_date, end_date, stride):
    """
    Apply same preprocessing pipeline and encode stocks using trained encoder
    """
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]
    vector_universe = {}
    
    for file_name in tqdm(files, desc="Processing stocks"):
        try:
            df = pd.read_csv(DIRECTORY + file_name, index_col=0)
            df = df[start_date:end_date]
            df_price = df['Close']
            
            # Need sufficient data for rolling z-score calculation and window generation
            min_required = LOOKBACK * 2 + stride  # Rolling calculation + windows + buffer
            if len(df_price) < min_required:
                print(f"⚠️  Skipping {file_name}: insufficient data ({len(df_price)} points, need {min_required})")
                continue
                
            print(f"📊 Processing {file_name}: {len(df_price)} data points")
            
            # Apply same preprocessing: rolling z-score
            rolling_mean = df_price.rolling(window=LOOKBACK).mean()
            rolling_std = df_price.rolling(window=LOOKBACK).std()
            rolling_zscore = (df_price - rolling_mean) / rolling_std
            rolling_zscore = rolling_zscore.dropna(how='all').to_list()
            
            # Ensure we have enough data after rolling calculation
            if len(rolling_zscore) < LOOKBACK + stride:
                print(f"⚠️  Skipping {file_name}: insufficient z-score data ({len(rolling_zscore)} points)")
                continue
            
            # Create windows - ensure we get exactly LOOKBACK-sized windows
            x = stride_window(stride, rolling_zscore, LOOKBACK)
            
            if len(x) == 0:
                print(f"⚠️  Skipping {file_name}: no valid windows generated")
                continue
            
            # Verify window dimensions
            print(f"🔍 Window shape for {file_name}: {x.shape}")
            if x.shape[1] != LOOKBACK:
                print(f"❌ Dimension mismatch for {file_name}: expected {LOOKBACK}, got {x.shape[1]}")
                continue
                
            # Encode using trained encoder
            stock_vect = encoder.predict(x).squeeze()
            stock_name = ''.join([letter for letter in file_name if letter.isupper()])
            vector_universe[stock_name] = stock_vect.tolist()
            print(f"✅ Successfully encoded {stock_name}: {len(stock_vect)} features")
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            continue
    
    return vector_universe

def cluster_pvals_multithreaded(vector_universe, start_date, end_date, n_stocks, alpha, max_workers=4):
    """
    Cluster stocks and find cointegrated N-tuples using multithreading
    Only tests cointegration WITHIN each cluster (not across clusters)
    """
    print(f"🔬 Clustering stocks and testing {n_stocks}-tuple cointegration (multithreaded)...")
    
    cluster_model = HDBSCAN(min_cluster_size=max(6, n_stocks + 1))
    data_labels = list(vector_universe.keys())
    vector_data = list(vector_universe.values())
    
    # Flatten and pad data
    flatten_to_2d = lambda data: [[i for sub in ([el] if not isinstance(el, list) else el) for i in (sub if not isinstance(sub, list) else [x for x in sub])] if not isinstance(item, list) else [i for sub in item for i in (sub if not isinstance(sub, list) else [x for x in sub])] for item in data]
    pad = lambda lst: [x + [0] * (max(map(len, lst)) - len(x)) for x in lst]
    vector_data = pad(flatten_to_2d(vector_data))
    
    cluster_set = cluster_model.fit_predict(vector_data).tolist()
    VECTOR_CLUSTER = dict(zip(data_labels, cluster_set))
    
    # Sort clusters
    sorted_clusters = {}
    for key, value in VECTOR_CLUSTER.items():
        if value not in sorted_clusters.keys():
            sorted_clusters[value] = [key]
        else:
            sorted_clusters[value].append(key)
    
    # Filter clusters that have enough stocks for N-tuples
    valid_clusters = {k: v for k, v in sorted_clusters.items() 
                     if len(v) >= n_stocks and k != -1}  # Exclude noise cluster (-1)
    
    print(f"📊 Found {len(valid_clusters)} valid clusters for {n_stocks}-tuple analysis")
    
    # Calculate total combinations across all clusters
    total_combos = sum(len(list(combinations(stocks, n_stocks))) 
                      for stocks in valid_clusters.values())
    print(f"🔍 Total combinations to test: {total_combos}")
    
    # Prepare file list once
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]
    
    # Process clusters in parallel
    all_results = []
    
    print(f"🚀 Starting multithreaded processing with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with common parameters
        process_cluster = partial(process_single_cluster, 
                                start_date=start_date, 
                                end_date=end_date, 
                                n_stocks=n_stocks, 
                                alpha=alpha, 
                                files=files)
        
        # Submit all cluster processing tasks
        future_to_cluster = {
            executor.submit(process_cluster, cluster_id, stock_list): cluster_id
            for cluster_id, stock_list in valid_clusters.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_cluster):
            cluster_id = future_to_cluster[future]
            try:
                cluster_results = future.result()
                all_results.extend(cluster_results)
            except Exception as e:
                print(f"❌ Error processing cluster {cluster_id}: {e}")
    
    print(f"✅ Multithreaded processing complete!")
    print(f"🎯 Found {len(all_results)} total cointegrated {n_stocks}-tuples across all clusters")
    
    return all_results

def calculate_spread(stock_tuple, start_date, end_date):
    """
    Calculate spread for an N-tuple of stocks using multiple regression
    """
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]
    
    # Load data for all stocks in tuple
    stock_data = {}
    for stock_name in stock_tuple:
        stock_file = [f for f in files if stock_name in f][0]
        df = pd.read_csv(DIRECTORY + stock_file, index_col=0)
        df = df[start_date:end_date]
        df.index = pd.to_datetime(df.index)
        stock_data[stock_name] = df['Close']
    
    # Align all dataframes
    combined_df = pd.concat(stock_data, axis=1).dropna()
    
    if len(stock_tuple) == 2:
        # Simple pair case
        stock1, stock2 = stock_tuple
        X = combined_df[stock2].values.reshape(-1, 1)
        y = combined_df[stock1].values
        reg = LinearRegression().fit(X, y)
        beta = reg.coef_[0]
        spread = combined_df[stock1] - beta * combined_df[stock2]
        coefficients = {stock1: 1.0, stock2: -beta}
    else:
        # Multiple regression for N-tuple
        dependent_stock = stock_tuple[0]  # First stock as dependent variable
        independent_stocks = stock_tuple[1:]  # Rest as independent variables
        
        X = combined_df[independent_stocks].values
        y = combined_df[dependent_stock].values
        reg = LinearRegression().fit(X, y)
        
        # Calculate spread: stock1 - β1*stock2 - β2*stock3 - ...
        spread = combined_df[dependent_stock].copy()
        coefficients = {dependent_stock: 1.0}
        
        for i, stock in enumerate(independent_stocks):
            spread -= reg.coef_[i] * combined_df[stock]
            coefficients[stock] = -reg.coef_[i]
    
    return combined_df, spread, coefficients

def create_modern_dashboard(top_tuples, initial_start_date, initial_end_date, n_stocks):
    """
    Create a modern, sleek interactive dashboard
    """
    print("🚀 Creating modern dashboard...")
    
    # Initialize Dash app with modern theme
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Modern sleek CSS
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <style>
                :root {
                    --primary-color: #2563eb;
                    --secondary-color: #1e40af;
                    --accent-color: #3b82f6;
                    --success-color: #10b981;
                    --warning-color: #f59e0b;
                    --danger-color: #ef4444;
                    --dark-bg: #0f172a;
                    --card-bg: #1e293b;
                    --border-color: #334155;
                    --text-primary: #f8fafc;
                    --text-secondary: #cbd5e1;
                    --shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                }
                
                body {
                    background: linear-gradient(135deg, var(--dark-bg) 0%, #1a202c 100%);
                    font-family: 'Inter', sans-serif;
                    color: var(--text-primary);
                    margin: 0;
                    min-height: 100vh;
                }
                
                .modern-card {
                    background: var(--card-bg);
                    border-radius: 16px;
                    border: 1px solid var(--border-color);
                    box-shadow: var(--shadow);
                    backdrop-filter: blur(16px);
                    margin: 16px;
                    padding: 24px;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }
                
                .modern-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                }
                
                .gradient-text {
                    background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    font-weight: 700;
                }
                
                .control-panel {
                    background: linear-gradient(135deg, var(--card-bg), #2d3748);
                }
                
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 16px;
                }
                
                .stat-item {
                    background: rgba(59, 130, 246, 0.1);
                    border: 1px solid rgba(59, 130, 246, 0.2);
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                }
                
                .stat-value {
                    font-size: 1.5rem;
                    font-weight: 600;
                    color: var(--accent-color);
                }
                
                .stat-label {
                    font-size: 0.875rem;
                    color: var(--text-secondary);
                    margin-top: 4px;
                }
                
                .modern-dropdown .Select-control {
                    background-color: var(--card-bg) !important;
                    border: 1px solid var(--border-color) !important;
                    border-radius: 8px !important;
                    color: var(--text-primary) !important;
                }
                
                .modern-button {
                    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                    border: none;
                    border-radius: 8px;
                    color: white;
                    padding: 12px 24px;
                    font-weight: 500;
                    transition: all 0.2s ease;
                    cursor: pointer;
                }
                
                .modern-button:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 10px 20px rgba(37, 99, 235, 0.3);
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Prepare dropdown options
    tuple_type = "Pairs" if n_stocks == 2 else f"{n_stocks}-tuples"
    tuple_options = [
        {'label': f"{' - '.join(tuple_data[0])} (p-val: {tuple_data[1]:.4f})", 
         'value': '|'.join(tuple_data[0])}
        for tuple_data in top_tuples
    ]
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1([
                    "🚀 AUTOENCODER ",
                    html.Span(f"{tuple_type.upper()} SCREENER", className="gradient-text")
                ], className="text-center mb-4", style={'fontSize': '2.5rem', 'fontWeight': '700'}),
                html.Hr(style={'border-color': 'var(--border-color)', 'margin': '2rem 0'}),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3([
                        "⚙️ ",
                        html.Span("CONTROL PANEL", className="gradient-text")
                    ], style={'fontSize': '1.5rem', 'marginBottom': '24px'}),
                    
                    # Screening Date Range
                    html.Label("📅 Screening Period", style={'color': 'var(--text-primary)', 'fontWeight': '500', 'marginBottom': '8px'}),
                    dcc.DatePickerRange(
                        id='screening-date-range',
                        start_date=initial_start_date,
                        end_date=initial_end_date,
                        display_format='YYYY-MM-DD',
                        style={'width': '100%', 'marginBottom': '16px'}
                    ),
                    
                    html.Button("🔍 Re-screen Combinations", 
                               id="rescreen-button", 
                               className="modern-button",
                               style={'width': '100%', 'marginBottom': '24px'}),
                    
                    # Graphing Date Range
                    html.Label("📊 Analysis Period", style={'color': 'var(--text-primary)', 'fontWeight': '500', 'marginBottom': '8px'}),
                    dcc.DatePickerRange(
                        id='graphing-date-range',
                        start_date=initial_start_date,
                        end_date=initial_end_date,
                        display_format='YYYY-MM-DD',
                        style={'width': '100%', 'marginBottom': '24px'}
                    ),
                    
                    html.Label(f"Select {tuple_type}", style={'color': 'var(--text-primary)', 'fontWeight': '500', 'marginBottom': '8px'}),
                    dcc.Dropdown(
                        id='tuple-selector',
                        options=tuple_options,
                        value=tuple_options[0]['value'] if tuple_options else None,
                        className='modern-dropdown',
                        style={'marginBottom': '24px'}
                    ),
                    
                    html.Label("Display Options", style={'color': 'var(--text-primary)', 'fontWeight': '500', 'marginBottom': '8px'}),
                    dcc.Checklist(
                        id='display-options',
                        options=[
                            {'label': ' Show Spread', 'value': 'spread'},
                            {'label': ' Show Individual Stocks', 'value': 'stocks'}
                        ],
                        value=['spread'],
                        style={'color': 'var(--text-secondary)', 'marginBottom': '24px'}
                    ),
                    
                    html.Label("Chart Theme", style={'color': 'var(--text-primary)', 'fontWeight': '500', 'marginBottom': '8px'}),
                    dcc.Dropdown(
                        id='theme-selector',
                        options=[
                            {'label': 'Modern Dark', 'value': 'modern_dark'},
                            {'label': 'Sleek Blue', 'value': 'sleek_blue'},
                            {'label': 'Minimal', 'value': 'minimal'},
                            {'label': 'Professional', 'value': 'professional'}
                        ],
                        value='modern_dark',
                        className='modern-dropdown'
                    )
                ], className="modern-card control-panel")
            ], width=3),
            
            dbc.Col([
                html.Div([
                    dcc.Graph(id='main-chart', style={'height': '600px'})
                ], className="modern-card")
            ], width=9)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4([
                        "📊 ",
                        html.Span("STATISTICS", className="gradient-text")
                    ], style={'fontSize': '1.25rem', 'marginBottom': '24px'}),
                    html.Div(id='stats-display')
                ], className="modern-card")
            ], width=12)
        ]),
        
        # Hidden div to store current tuples data
        html.Div(id='tuples-data', style={'display': 'none'})
    ], fluid=True, style={'padding': '20px'})
    
    @app.callback(
        [Output('tuples-data', 'children'),
         Output('tuple-selector', 'options'),
         Output('tuple-selector', 'value')],
        [Input('rescreen-button', 'n_clicks')],
        [dash.dependencies.State('screening-date-range', 'start_date'),
         dash.dependencies.State('screening-date-range', 'end_date')]
    )
    def rescreen_tuples(n_clicks, start_date, end_date):
        if n_clicks and start_date and end_date:
            print(f"🔄 Re-screening {tuple_type} for period {start_date} to {end_date}")
            
            # Load encoder
            encoder = load_encoder()
            if encoder is None:
                return str(top_tuples), tuple_options, tuple_options[0]['value'] if tuple_options else None
            
            # Re-screen with new dates using multithreaded approach
            new_tuples = screen_stocks(encoder, start_date, end_date, n_stocks=n_stocks, alpha=0.05)
            
            if new_tuples:
                new_tuple_options = [
                    {'label': f"{' - '.join(tuple_data[0])} (p-val: {tuple_data[1]:.4f})", 
                     'value': '|'.join(tuple_data[0])}
                    for tuple_data in new_tuples[:10]
                ]
                return str(new_tuples), new_tuple_options, new_tuple_options[0]['value']
        
        return str(top_tuples), tuple_options, tuple_options[0]['value'] if tuple_options else None
    
    @app.callback(
        [Output('main-chart', 'figure'),
         Output('stats-display', 'children')],
        [Input('tuple-selector', 'value'),
         Input('display-options', 'value'),
         Input('theme-selector', 'value'),
         Input('graphing-date-range', 'start_date'),
         Input('graphing-date-range', 'end_date'),
         Input('tuples-data', 'children')]
    )
    def update_dashboard(selected_tuple, display_options, theme_style, graph_start, graph_end, tuples_data):
        if not selected_tuple or not graph_start or not graph_end:
            return {}, "No combination selected or invalid date range"
        
        stock_tuple = tuple(selected_tuple.split('|'))
        
        try:
            # Calculate spread for graphing period
            combined_df, spread, coefficients = calculate_spread(stock_tuple, graph_start, graph_end)
            
            # Calculate p-value for the graphing period
            asset_data = {stock: combined_df[stock].tolist() for stock in stock_tuple}
            current_p_val = multi_cointegration_test(asset_data)['p_value']
            
            # Theme styles
            themes = {
                'modern_dark': {
                    'plot_bgcolor': '#0f172a',
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'axis_color': 'rgba(59, 130, 246, 0.1)',
                    'colors': ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
                },
                'sleek_blue': {
                    'plot_bgcolor': '#1e293b',
                    'paper_bgcolor': 'rgba(30, 41, 59, 0.8)',
                    'axis_color': 'rgba(59, 130, 246, 0.2)',
                    'colors': ['#60a5fa', '#34d399', '#fbbf24', '#f87171', '#a78bfa']
                },
                'minimal': {
                    'plot_bgcolor': '#f8fafc',
                    'paper_bgcolor': 'rgba(248, 250, 252, 0.9)',
                    'axis_color': 'rgba(148, 163, 184, 0.3)',
                    'colors': ['#1e40af', '#dc2626', '#059669', '#d97706', '#7c3aed']
                },
                'professional': {
                    'plot_bgcolor': '#111827',
                    'paper_bgcolor': 'rgba(17, 24, 39, 0.95)',
                    'axis_color': 'rgba(75, 85, 99, 0.3)',
                    'colors': ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed']
                }
            }
            
            current_theme = themes[theme_style]
            
            # Create subplots
            subplot_count = len([opt for opt in display_options if opt in ['spread', 'stocks']])
            if subplot_count == 0:
                subplot_count = 1
                
            fig = make_subplots(
                rows=subplot_count, cols=1,
                subplot_titles=[],
                vertical_spacing=0.08
            )
            
            current_row = 1
            color_idx = 0
            
            # Add traces based on display options
            if 'stocks' in display_options:
                for stock in stock_tuple:
                    fig.add_trace(
                        go.Scatter(
                            x=combined_df.index, 
                            y=combined_df[stock], 
                            name=stock, 
                            line=dict(color=current_theme['colors'][color_idx % len(current_theme['colors'])], width=2)
                        ),
                        row=current_row, col=1
                    )
                    color_idx += 1
                current_row += 1
            
            if 'spread' in display_options:
                # Create spread equation string
                spread_eq = f"{stock_tuple[0]}"
                for i, stock in enumerate(stock_tuple[1:], 1):
                    coef = coefficients[stock]
                    spread_eq += f" {'+' if coef >= 0 else '-'} {abs(coef):.3f}*{stock}"
                
                fig.add_trace(
                    go.Scatter(
                        x=spread.index, 
                        y=spread, 
                        name=f'Spread ({spread_eq})', 
                        line=dict(color=current_theme['colors'][color_idx % len(current_theme['colors'])], width=3)
                    ),
                    row=current_row, col=1
                )
                # Add zero line for spread
                fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", 
                             row=current_row, col=1, opacity=0.6)
            
            # Update layout with modern styling
            fig.update_layout(
                title=dict(
                    text=f"<b>📈 {' × '.join(stock_tuple)} Analysis | P-VAL: {current_p_val:.6f}</b>",
                    font=dict(color='var(--text-primary)', size=20, family='Inter'),
                    x=0.5
                ),
                **current_theme,
                font=dict(color='var(--text-primary)', family='Inter'),
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(30, 41, 59, 0.8)',
                    bordercolor='var(--border-color)',
                    borderwidth=1,
                    font=dict(color='var(--text-primary)')
                ),
                xaxis=dict(
                    gridcolor=current_theme['axis_color'],
                    color='var(--text-primary)',
                    showgrid=True
                ),
                yaxis=dict(
                    gridcolor=current_theme['axis_color'],
                    color='var(--text-primary)',
                    showgrid=True
                ),
                hovermode='x unified'
            )
            
            # Update all subplot axes
            for i in range(1, subplot_count + 1):
                fig.update_xaxes(
                    gridcolor=current_theme['axis_color'],
                    color='var(--text-primary)',
                    showgrid=True,
                    row=i, col=1
                )
                fig.update_yaxes(
                    gridcolor=current_theme['axis_color'],
                    color='var(--text-primary)',
                    showgrid=True,
                    row=i, col=1
                )
            
            # Statistics display
            screening_p_val = None
            try:
                current_tuples = eval(tuples_data) if tuples_data else top_tuples
                screening_p_val = next((tuple_data[1] for tuple_data in current_tuples 
                                      if '|'.join(tuple_data[0]) == selected_tuple), None)
            except:
                screening_p_val = None
            
            stats = html.Div([
                html.Div([
                    html.Div([
                        html.Div(f"{len(stock_tuple)}", className="stat-value"),
                        html.Div("Stocks", className="stat-label")
                    ], className="stat-item"),
                    
                    html.Div([
                        html.Div(f"{current_p_val:.6f}", className="stat-value"),
                        html.Div("Current P-Value", className="stat-label")
                    ], className="stat-item"),
                    
                    html.Div([
                        html.Div(f"{screening_p_val:.6f}" if screening_p_val else "N/A", className="stat-value"),
                        html.Div("Screening P-Value", className="stat-label")
                    ], className="stat-item"),
                    
                    html.Div([
                        html.Div(f"{spread.mean():.4f}", className="stat-value"),
                        html.Div("Spread Mean", className="stat-label")
                    ], className="stat-item"),
                    
                    html.Div([
                        html.Div(f"{spread.std():.4f}", className="stat-value"),
                        html.Div("Spread Std", className="stat-label")
                    ], className="stat-item"),
                    
                    html.Div([
                        html.Div(f"{len(combined_df)}", className="stat-value"),
                        html.Div("Data Points", className="stat-label")
                    ], className="stat-item"),
                    
                    html.Div([
                        html.Div("✅ YES" if current_p_val < 0.05 else "❌ NO", className="stat-value"),
                        html.Div("Cointegrated", className="stat-label")
                    ], className="stat-item")
                ], className="stats-grid")
            ])
            
            return fig, stats
            
        except Exception as e:
            error_fig = go.Figure()
            error_fig.update_layout(
                title=f"Error: {str(e)}",
                plot_bgcolor='#0f172a',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return error_fig, f"Error processing combination: {str(e)}"
    
    return app

def main():
    """
    Main screening application
    """
    print("🚀 AUTOENCODER STOCK COMBINATION SCREENER INITIATED")
    print("=" * 60)
    
    # Get user preferences
    n_stocks = get_user_preferences()
    
    # Load trained encoder
    encoder = load_encoder()
    if encoder is None:
        return
    
    # Configuration - Using wider date range to ensure sufficient data for 256-dimensional windows
    start_date = '2019-01-01'  # Wider window to ensure sufficient data for LOOKBACK=256
    end_date = '2021-12-31'    # Longer period to get more data points
    alpha = 0.05
    
    print(f"⏰ Screening period: {start_date} to {end_date}")
    print(f"🎯 Alpha threshold: {alpha}")
    print(f"🔧 Expected input dimension: {LOOKBACK}")
    
    # Screen for cointegrated tuples
    top_tuples = screen_stocks(encoder, start_date, end_date, n_stocks=n_stocks, alpha=alpha)
    
    if not top_tuples:
        print("❌ No cointegrated combinations found. Try increasing alpha or checking data availability.")
        return
    
    tuple_type = "pairs" if n_stocks == 2 else f"{n_stocks}-tuples"
    print(f"\n🏆 TOP COINTEGRATED {tuple_type.upper()}:")
    print("-" * 50)
    for i, (stocks, p_val) in enumerate(top_tuples[:10], 1):
        print(f"{i:2d}. {' × '.join(stocks)}: p-value = {p_val:.6f}")
    
    # Create and run dashboard
    app = create_modern_dashboard(top_tuples[:10], start_date, end_date, n_stocks)
    
    print(f"\n🚀 Launching modern {tuple_type} dashboard...")
    print("📡 Access at: http://127.0.0.1:8050")
    print("🛑 Press Ctrl+C to stop")
    
    # Use app.run() instead of app.run_server()
    app.run(debug=True, host='127.0.0.1', port=8050)

if __name__ == "__main__":
    main() 