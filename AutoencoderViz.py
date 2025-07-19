import pickle
import random 
from datetime import datetime, timedelta
import pandas as pd
import os
import numpy as np
from itertools import combinations
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from sklearn.cluster import HDBSCAN
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import logging
import warnings
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF debugging logs
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress specific TensorFlow messages
tf.get_logger().setLevel('ERROR')

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"🚀 {title}")
    print("="*80)

# Configuration Constants
BACKGROUND_COLOR = 'black'  # Change this to modify dashboard background
# Try different possible paths for stock data
possible_paths = [
    'stock_data/Data/',
    'CPCV-Ensemble-ML-for-Pairs-Spread/stock_data/Data/',
    './stock_data/Data/'
]
DIRECTORY = None
for path in possible_paths:
    if os.path.exists(path):
        DIRECTORY = path
        break

if DIRECTORY is None:
    print("Error: Could not find stock data directory")
    print("Looked in:", possible_paths)

LOOKBACK = 256  # Must match the training lookback
STRIDE = 64     # Must match the training stride

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

def stride_window(stride, lst, window):
    """
    stride: distance between each window
    window: size per sample 
    lst   : 1D list to apply stride to
    returns: 2D array with windowed samples
    """
    sample_amt = ((len(lst) - window) / stride) + 1
    wrap_backwards = False 
    result = []
    start  = 0
    end    = window  
    if isinstance(sample_amt, float):  # Not even overlap
        wrap_backwards = True
    for i in range(int(sample_amt)):
        result.append(lst[start:end]) 
        start += stride
        end   += stride
    if wrap_backwards:
        result.append(lst[-window:])
    return np.array(result)

def load_encoder():
    """Load the saved encoder model"""
    try:
        with open("encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        return encoder
    except FileNotFoundError:
        # Try alternative path
        try:
            with open("CPCV-Ensemble-ML-for-Pairs-Spread/encoder.pkl", "rb") as f:
                encoder = pickle.load(f)
            return encoder
        except FileNotFoundError:
            print("Error: encoder.pkl not found. Please run AutoEncoderPairScreenerV2.py first.")
            print("Looking for encoder.pkl in current directory and CPCV-Ensemble-ML-for-Pairs-Spread/")
            return None

def encode_stock_data(encoder, start_date, end_date):
    """
    Encode stock data using the saved encoder
    Returns vector_universe: dict of stock_name: encoded_vectors
    """
    print_section_header("LOADING AND ENCODING STOCK DATA")
    print(f"📅 Analysis Period: {start_date} to {end_date}")
    
    if not os.path.exists(DIRECTORY):
        logging.error(f"Directory {DIRECTORY} not found.")
        print("❌ Error: Stock data directory not found!")
        print(f"   Looked in: {DIRECTORY}")
        return {}
    
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]
    total_files = len(files)
    print(f"\n📊 Found {total_files} stock files to process")
    
    vector_universe = {}
    processed = 0
    skipped = 0
    errors = 0
    
    for file_name in files:
        try:
            processed += 1
            print(f"\n🔄 Processing {file_name} ({processed}/{total_files})")
            
            df = pd.read_csv(os.path.join(DIRECTORY, file_name), index_col=0)
            df = df[start_date:end_date]
            df_price = df['Close']
            
            if len(df_price) < LOOKBACK * 3:
                print(f"⚠️  Skipping {file_name}: Insufficient data ({len(df_price)} points, need {LOOKBACK * 3})")
                skipped += 1
                continue
            
            # Rolling z-score calculation
            print(f"   📈 Calculating rolling statistics...")
            rolling_mean = df_price.rolling(window=LOOKBACK).mean()
            rolling_std = df_price.rolling(window=LOOKBACK).std()
            rolling_zscore = (df_price - rolling_mean) / rolling_std
            rolling_zscore = rolling_zscore.dropna(how='all').to_list()
            
            print(f"   🪟 Creating sliding windows...")
            x = stride_window(STRIDE, rolling_zscore, LOOKBACK)
            
            print(f"   🧠 Encoding with autoencoder...")
            stock_name = ''.join([letter for letter in file_name if letter.isupper()])
            stock_vect = encoder.predict(x, verbose=0).squeeze()
            vector_universe[stock_name] = stock_vect.tolist()
            
            print(f"   ✅ Successfully encoded {stock_name}")
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {str(e)}")
            errors += 1
            continue
    
    print("\n📊 Encoding Summary:")
    print(f"   ✅ Successfully processed: {len(vector_universe)} stocks")
    print(f"   ⚠️  Skipped (insufficient data): {skipped} stocks")
    print(f"   ❌ Errors: {errors} stocks")
    print(f"   📈 Total processed: {processed} stocks")
    
    return vector_universe

def cluster_and_test_pairs(vector_universe, start_date, end_date, alpha=0.1, include_triplets=False):
    """
    Cluster encoded vectors and test cointegration for pairs (and optionally triplets)
    Returns: (pairs_with_pvals, cluster_info)
    """
    print_section_header("CLUSTERING AND COINTEGRATION ANALYSIS")
    
    if not vector_universe:
        print("❌ No encoded stock data available!")
        return [], {}
    
    print("🔍 Preparing data for clustering...")
    
    # More aggressive clustering parameters to get more clusters
    cluster_model = HDBSCAN(
        min_cluster_size=3,  # Reduced from 6 to 3
        min_samples=2,       # Minimum samples in a neighborhood
        cluster_selection_epsilon=0.1,  # Distance threshold
        alpha=1.0           # Controls how conservative the clustering is
    )
    
    data_labels = list(vector_universe.keys())
    vector_data = list(vector_universe.values())
    
    print("📊 Flattening and padding vectors...")
    flatten_to_2d = lambda data: [[i for sub in ([el] if not isinstance(el, list) else el) for i in (sub if not isinstance(sub, list) else [x for x in sub])] if not isinstance(item, list) else [i for sub in item for i in (sub if not isinstance(sub, list) else [x for x in sub])] for item in data]
    pad = lambda lst: [x + [0] * (max(map(len, lst)) - len(x)) for x in lst]
    vector_data = pad(flatten_to_2d(vector_data))
    
    print(f"📏 Data shape: {len(vector_data)} stocks × {len(vector_data[0])} features")
    
    print("\n🧮 Running HDBSCAN clustering...")
    cluster_set = cluster_model.fit_predict(vector_data).tolist()
    
    VECTOR_CLUSTER = dict(zip(data_labels, cluster_set))
    sorted_clusters = {}
    combo_universe = []
    
    # Sort clusters
    for key, value in VECTOR_CLUSTER.items():
        if value not in sorted_clusters.keys():
            sorted_clusters[value] = [key]
        else:
            sorted_clusters[value].append(key)
    
    # Print cluster information
    print("\n" + "="*60)
    print("🔍 CLUSTER ANALYSIS RESULTS")
    print("="*60)
    
    total_clustered = 0
    valid_clusters = 0
    
    for cluster_id in sorted(sorted_clusters.keys()):
        stocks = sorted_clusters[cluster_id]
        if cluster_id == -1:
            print(f"\n🔸 Noise (Cluster {cluster_id}): {len(stocks)} stocks")
            print(f"   Stocks: {', '.join(stocks[:10])}{'...' if len(stocks) > 10 else ''}")
        else:
            print(f"\n📊 Cluster {cluster_id}: {len(stocks)} stocks")
            print(f"   Stocks: {', '.join(stocks)}")
            total_clustered += len(stocks)
            valid_clusters += 1
    
    print(f"\n📈 Clustering Summary:")
    print(f"   • Total valid clusters: {valid_clusters}")
    print(f"   • Total stocks in clusters: {total_clustered}")
    print(f"   • Noise points: {len(sorted_clusters.get(-1, []))}")
    print(f"   • Total stocks processed: {len(data_labels)}")
    print("="*60)
    
    # Generate combinations - include pairs from ALL clusters (even noise points)
    print("\n🔄 Generating stock combinations...")
    pairs_from_clusters = 0
    pairs_from_noise = 0
    
    for key, lst in sorted_clusters.items():
        if len(lst) >= 2:
            # Generate pairs
            pairs = list(combinations(lst, 2))
            for item in pairs:
                combo_universe.append(item)
            
            if key == -1:
                pairs_from_noise += len(pairs)
                print(f"   🔸 Generated {len(pairs)} pairs from noise points")
            else:
                pairs_from_clusters += len(pairs)
                print(f"   📊 Generated {len(pairs)} pairs from cluster {key}")
            
            # Generate triplets if requested
            if include_triplets and len(lst) >= 3:
                triplets = list(combinations(lst, 3))
                for item in triplets:
                    combo_universe.append(item)
                print(f"   📈 Generated {len(triplets)} triplets from cluster {key}")
    
    print(f"\n📊 Combination Summary:")
    print(f"   • Pairs from clusters: {pairs_from_clusters}")
    print(f"   • Pairs from noise: {pairs_from_noise}")
    print(f"   • Total combinations: {len(combo_universe)}")
    
    print(f"\n🧪 Testing {len(combo_universe)} combinations for cointegration...")
    print(f"   Alpha threshold: {alpha}")
    
    # Test cointegration
    final_p_vals = {}
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]
    
    success = 0
    failed = 0
    
    for assets in tqdm(combo_universe, desc="Testing cointegration"):
        try:
            asset_data = {} 
            for name in assets:
                matching_files = [f for f in files if name in f]
                if not matching_files:
                    continue
                df = pd.read_csv(os.path.join(DIRECTORY, matching_files[0]), index_col=0)
                df = df[start_date:end_date] 
                df.index = pd.to_datetime(df.index)
                asset_data[name] = df.Close.to_list() 
            
            if len(asset_data) >= 2:
                result = multi_cointegration_test(asset_data)
                if len(asset_data) == 2:
                    p_val = result['p_value']
                else:
                    p_val = 0.05 if result['cointegration_rank'] > 0 else 1.0
                final_p_vals[assets] = p_val
                success += 1
        except Exception as e:
            print(f"\n❌ Error testing {assets}: {str(e)}")
            failed += 1
            continue
    
    top_pairs = [(item[0], item[1]) for item in final_p_vals.items() if item[1] <= alpha]
    
    print("\n📊 Cointegration Testing Results:")
    print(f"   ✅ Successfully tested: {success} combinations")
    print(f"   ❌ Failed tests: {failed} combinations")
    print(f"   🎯 Found {len(top_pairs)} cointegrated pairs/triplets with p-value ≤ {alpha}")
    
    if len(top_pairs) > 0:
        print(f"\n🏆 Top 5 best pairs:")
        sorted_pairs = sorted(top_pairs, key=lambda x: x[1])
        for i, (pair, p_val) in enumerate(sorted_pairs[:5]):
            if len(pair) == 2:
                print(f"   {i+1}. {pair[0]} ↔ {pair[1]} (p={p_val:.6f})")
            else:
                print(f"   {i+1}. {' ↔ '.join(pair)} (p={p_val:.6f})")
    
    # Prepare cluster info for visualization
    cluster_info = {
        'clusters': sorted_clusters,
        'vector_data': vector_data,
        'labels': data_labels,
        'cluster_assignments': cluster_set
    }
    
    return top_pairs, cluster_info

def get_pair_data(assets, start_date, end_date):
    """Get price data for multiple assets"""
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]
    asset_data = {}
    
    for stock in assets:
        matching_files = [f for f in files if stock in f]
        if matching_files:
            df = pd.read_csv(os.path.join(DIRECTORY, matching_files[0]), index_col=0)
            df = df[start_date:end_date]
            df.index = pd.to_datetime(df.index)
            asset_data[stock] = df['Close']
    
    return asset_data

def calculate_spread_and_stats(pair_data, split_ratio=0.2):
    """Calculate spread and statistics for a pair"""
    if len(pair_data) != 2:
        return None
    
    stocks = list(pair_data.keys())
    stock1_data = pair_data[stocks[0]]
    stock2_data = pair_data[stocks[1]]
    
    # Align data
    aligned_data = pd.concat([stock1_data, stock2_data], axis=1).dropna()
    aligned_data.columns = stocks
    
    if len(aligned_data) == 0:
        return None
    
    # Calculate beta using first 20% of data
    split_point = int(len(aligned_data) * split_ratio)
    train_data = aligned_data.iloc[:split_point]
    
    # Linear regression for beta calculation (spread)
    X_train = train_data[stocks[1]].values.reshape(-1, 1)
    y_train = train_data[stocks[0]].values
    lr_spread = LinearRegression().fit(X_train, y_train)
    beta_spread = lr_spread.coef_[0]
    
    # Calculate spread for entire period
    spread = aligned_data[stocks[0]] - beta_spread * aligned_data[stocks[1]]
    
    # Linear regression for entire dataset (for 2D plot)
    X_full = aligned_data[stocks[1]].values.reshape(-1, 1)
    y_full = aligned_data[stocks[0]].values
    lr_full = LinearRegression().fit(X_full, y_full)
    
    # Calculate correlation and other stats
    correlation = aligned_data[stocks[0]].corr(aligned_data[stocks[1]])
    
    return {
        'aligned_data': aligned_data,
        'spread': spread,
        'beta_spread': beta_spread,
        'lr_full': lr_full,
        'correlation': correlation,
        'stocks': stocks
    }

def create_styled_figure():
    """Create a standardized plotly figure with consistent styling"""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,20,0.8)',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color='#FFD700'
        ),
        title_font=dict(
            size=16,
            color='#FFD700'
        ),
        margin=dict(l=60, r=60, t=60, b=60),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='#FFD700',
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor='rgba(255,215,0,0.2)',
            zerolinecolor='rgba(255,215,0,0.3)',
            color='#FFD700'
        ),
        yaxis=dict(
            gridcolor='rgba(255,215,0,0.2)',
            zerolinecolor='rgba(255,215,0,0.3)',
            color='#FFD700'
        )
    )
    return fig

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load encoder and initial data
encoder = load_encoder()
if encoder is None:
    print("Cannot proceed without encoder. Exiting.")
    exit(1)

# Define enhanced color scheme
colors = {
    'background': BACKGROUND_COLOR,
    'text': '#FFD700',  # Gold/Yellow
    'secondary': '#1a1a1a',
    'accent': '#FFA500',  # Orange
    'success': '#00FF7F',  # Spring Green
    'warning': '#FF6347',   # Tomato
    'info': '#40E0D0',     # Turquoise
    'card_bg': 'rgba(26,26,26,0.9)',
    'border': 'rgba(255,215,0,0.3)'
}

# Custom CSS styling
custom_style = {
    'backgroundColor': colors['background'],
    'minHeight': '100vh',
    'fontFamily': 'Arial, sans-serif'
}

card_style = {
    'backgroundColor': colors['card_bg'],
    'border': f'1px solid {colors["border"]}',
    'borderRadius': '10px',
    'padding': '20px',
    'margin': '10px 0',
    'boxShadow': '0 4px 8px rgba(255,215,0,0.1)'
}

# App layout with tabs for better organization
app.layout = dbc.Container([
    # Hidden data store for pairs
    dcc.Store(id='pairs-store', data=[]),
    dcc.Store(id='analysis-complete', data=False),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("⚡ Autoencoder Pairs Trading Dashboard", 
                       style={
                           'textAlign': 'center', 
                           'color': colors['text'], 
                           'marginBottom': '10px',
                           'fontSize': '2.5rem',
                           'fontWeight': 'bold'
                       }),
                html.P("AI-Powered Statistical Arbitrage Analysis", 
                      style={
                          'textAlign': 'center', 
                          'color': colors['accent'], 
                          'fontSize': '1.2rem',
                          'marginBottom': '30px'
                      })
            ], style=card_style)
        ])
    ]),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("🎛️ Analysis Parameters", style={'color': colors['text'], 'marginBottom': '20px'}),
                dbc.Row([
                    dbc.Col([
                        html.Label("Start Date", style={'color': colors['text'], 'fontWeight': 'bold'}),
                        dcc.DatePickerSingle(
                            id='start-date',
                            date='2014-01-01',
                            style={'width': '100%'}
                        )
                    ], width=2),
                    dbc.Col([
                        html.Label("End Date", style={'color': colors['text'], 'fontWeight': 'bold'}),
                        dcc.DatePickerSingle(
                            id='end-date',
                            date='2020-01-01',
                            style={'width': '100%'}
                        )
                    ], width=2),
                    dbc.Col([
                        html.Label("Alpha Threshold", style={'color': colors['text'], 'fontWeight': 'bold'}),
                        dcc.Input(
                            id='alpha-input',
                            type='number',
                            value=0.1,
                            min=0.01,
                            max=1.0,
                            step=0.01,
                            style={'width': '100%', 'backgroundColor': colors['secondary'], 'color': colors['text']}
                        )
                    ], width=2),
                    dbc.Col([
                        html.Label("Triplets", style={'color': colors['text'], 'fontWeight': 'bold'}),
                        dcc.Checklist(
                            id='triplets-checkbox',
                            options=[{'label': ' Enable', 'value': 'enable'}],
                            value=[],
                            style={'color': colors['text'], 'marginTop': '8px'}
                        )
                    ], width=2),
                    dbc.Col([
                        dbc.Button("🚀 Run Analysis", id="update-btn", color="warning", size="lg",
                                  style={'marginTop': '25px', 'width': '100%', 'fontWeight': 'bold'})
                    ], width=4)
                ])
            ], style=card_style)
        ])
    ], style={'marginBottom': '20px'}),
    
    # Status indicator
    dbc.Row([
        dbc.Col([
            html.Div(id="status-indicator", style={'textAlign': 'center', 'marginBottom': '20px'})
        ])
    ]),
    
    # Main Content with Tabs
    dbc.Row([
        dbc.Col([
            dcc.Tabs(
                id="main-tabs",
                value="clusters",
                children=[
                    dcc.Tab(
                        label="📊 Cluster Analysis",
                        value="clusters",
                        style={'backgroundColor': colors['secondary'], 'color': colors['text']},
                        selected_style={'backgroundColor': colors['accent'], 'color': 'black'}
                    ),
                    dcc.Tab(
                        label="💰 Pairs Analysis", 
                        value="pairs",
                        style={'backgroundColor': colors['secondary'], 'color': colors['text']},
                        selected_style={'backgroundColor': colors['accent'], 'color': 'black'}
                    )
                ],
                style={'height': '50px'}
            ),
            html.Div(id="tab-content", style={'marginTop': '20px'})
        ])
    ])
    
], fluid=True, style=custom_style)

# Update analysis callback - stores data in dcc.Store
@callback(
    [Output('pairs-store', 'data'),
     Output('analysis-complete', 'data'),
     Output('status-indicator', 'children')],
    [Input('update-btn', 'n_clicks')],
    [dash.State('start-date', 'date'),
     dash.State('end-date', 'date'),
     dash.State('alpha-input', 'value'),
     dash.State('triplets-checkbox', 'value')]
)
def run_analysis(n_clicks, start_date, end_date, alpha, triplets_enabled):
    if not n_clicks or n_clicks == 0:
        return [], False, html.P("Click 'Run Analysis' to start", style={'color': colors['text']})
    
    # Show loading status
    status = dbc.Alert([
        dbc.Spinner(color="warning", size="sm"),
        " Running analysis... This may take several minutes."
    ], color="info", style={'textAlign': 'center'})
    
    try:
        include_triplets = 'enable' in triplets_enabled if triplets_enabled else False
        
        # Encode stock data
        vector_universe = encode_stock_data(encoder, start_date, end_date)
        
        # Cluster and test pairs
        pairs_with_pvals, cluster_info = cluster_and_test_pairs(vector_universe, start_date, end_date, alpha, include_triplets)
        
        # Prepare pairs data for storage
        pairs_data = {
            'pairs_with_pvals': pairs_with_pvals,
            'cluster_info': cluster_info,
            'start_date': start_date,
            'end_date': end_date,
            'alpha': alpha
        }
        
        success_status = dbc.Alert([
            html.I(className="fas fa-check-circle"),
            f" Analysis complete! Found {len(pairs_with_pvals)} cointegrated pairs."
        ], color="success", style={'textAlign': 'center'})
        
        return pairs_data, True, success_status
        
    except Exception as e:
        error_status = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle"),
            f" Analysis failed: {str(e)}"
        ], color="danger", style={'textAlign': 'center'})
        
        return [], False, error_status

# Tab content callback
@callback(
    Output("tab-content", "children"),
    [Input("main-tabs", "value"),
     Input('pairs-store', 'data'),
     Input('analysis-complete', 'data')]
)
def render_tab_content(active_tab, pairs_data, analysis_complete):
    if active_tab == "clusters":
        if not analysis_complete or not pairs_data:
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("🎯 Cluster Information", style={'color': colors['text']}),
                            html.P("Run analysis to see cluster information", style={'color': colors['text'], 'textAlign': 'center'})
                        ], style=card_style)
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("🗺️ Cluster Visualization (PCA)", style={'color': colors['text']}),
                            dcc.Graph(figure=create_styled_figure(), style={'height': '500px'})
                        ], style=card_style)
                    ], width=12)
                ])
            ])
        
        # Generate cluster information display
        cluster_info = pairs_data['cluster_info']
        clusters = cluster_info.get('clusters', {})
        pairs_with_pvals = pairs_data['pairs_with_pvals']
        
        cluster_info_div = []
        
        total_pairs = len([pair for pair in pairs_with_pvals if len(pair[0]) == 2])
        total_triplets = len([pair for pair in pairs_with_pvals if len(pair[0]) == 3])
        
        # Summary card
        cluster_info_div.append(
            html.Div([
                html.H5("📊 Analysis Summary", style={'color': colors['success'], 'marginBottom': '15px'}),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3(f"{len([c for c in clusters.keys() if c != -1])}", style={'color': colors['accent'], 'margin': 0}),
                            html.P("Valid Clusters", style={'color': colors['text'], 'margin': 0})
                        ], style={'textAlign': 'center'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H3(f"{total_pairs}", style={'color': colors['success'], 'margin': 0}),
                            html.P("Cointegrated Pairs", style={'color': colors['text'], 'margin': 0})
                        ], style={'textAlign': 'center'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H3(f"{total_triplets}", style={'color': colors['info'], 'margin': 0}),
                            html.P("Cointegrated Triplets", style={'color': colors['text'], 'margin': 0})
                        ], style={'textAlign': 'center'})
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.H3(f"{len(clusters.get(-1, []))}", style={'color': colors['warning'], 'margin': 0}),
                            html.P("Noise Points", style={'color': colors['text'], 'margin': 0})
                        ], style={'textAlign': 'center'})
                    ], width=3)
                ])
            ], style={'backgroundColor': 'rgba(0,255,127,0.1)', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'})
        )
        
        # Individual cluster cards
        for cluster_id in sorted([c for c in clusters.keys() if c != -1]):
            stocks = clusters[cluster_id]
            cluster_info_div.append(
                html.Div([
                    html.H6(f"🎯 Cluster {cluster_id}", style={'color': colors['accent'], 'marginBottom': '10px'}),
                    html.P(f"📊 {len(stocks)} stocks", style={'color': colors['text'], 'marginBottom': '5px'}),
                    html.P(f"📈 {', '.join(stocks)}", style={'color': colors['info'], 'fontSize': '0.9rem'})
                ], style={
                    'backgroundColor': 'rgba(255,165,0,0.1)', 
                    'padding': '15px', 
                    'borderRadius': '8px', 
                    'marginBottom': '10px',
                    'border': f'1px solid {colors["accent"]}'
                })
            )
        
        # Generate cluster visualization
        cluster_fig = create_styled_figure()
        
        if cluster_info.get('vector_data'):
            # Apply PCA for 2D visualization
            pca = PCA(n_components=2)
            vector_2d = pca.fit_transform(cluster_info['vector_data'])
            
            # Color by cluster with enhanced palette
            unique_clusters = set(cluster_info['cluster_assignments'])
            colors_list = px.colors.qualitative.Set3
            
            for i, cluster_id in enumerate(unique_clusters):
                mask = [c == cluster_id for c in cluster_info['cluster_assignments']]
                cluster_data = vector_2d[np.array(mask)]
                cluster_labels = [cluster_info['labels'][j] for j, m in enumerate(mask) if m]
                
                color = colors_list[i % len(colors_list)] if cluster_id != -1 else '#666666'
                name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
                
                cluster_fig.add_trace(go.Scatter(
                    x=cluster_data[:, 0],
                    y=cluster_data[:, 1],
                    mode='markers+text',
                    text=cluster_labels,
                    textposition='top center',
                    textfont=dict(size=8, color='white'),
                    marker=dict(
                        color=color, 
                        size=12,
                        line=dict(width=1, color='white'),
                        opacity=0.8
                    ),
                    name=name,
                    hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
                ))
        
        cluster_fig.update_layout(
            title=dict(
                text="Stock Clusters in Latent Space (PCA Projection)",
                x=0.5,
                font=dict(size=18, color=colors['text'])
            ),
            xaxis_title="First Principal Component",
            yaxis_title="Second Principal Component",
            height=500
        )
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("🎯 Cluster Information", style={'color': colors['text']}),
                        html.Div(cluster_info_div)
                    ], style=card_style)
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("🗺️ Cluster Visualization (PCA)", style={'color': colors['text']}),
                        dcc.Graph(figure=cluster_fig, style={'height': '500px'})
                    ], style=card_style)
                ], width=12)
            ])
        ])
    
    elif active_tab == "pairs":
        if not analysis_complete or not pairs_data:
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("📋 Select Trading Pair", style={'color': colors['text']}),
                            html.P("Run analysis first to see available pairs", style={'color': colors['text'], 'textAlign': 'center'})
                        ], style=card_style)
                    ], width=12)
                ])
            ])
        
        # Prepare dropdown options
        pairs_with_pvals = pairs_data['pairs_with_pvals']
        pair_options = []
        
        for pair in pairs_with_pvals:
            assets = pair[0]
            p_val = pair[1]
            if len(assets) == 2:
                label = f"🔗 {assets[0]} ↔ {assets[1]} (p={p_val:.4f})"
                value = f"{assets[0]}|{assets[1]}"
            else:  # Triplets
                label = f"🔺 {' ↔ '.join(assets)} (p={p_val:.4f})"
                value = "|".join(assets)
            pair_options.append({'label': label, 'value': value})
        
        return html.Div([
            # Pair Selection and Analysis Date Range
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("📋 Pair Selection & Analysis Period", style={'color': colors['text']}),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Trading Pair", style={'color': colors['text'], 'fontWeight': 'bold'}),
                                dcc.Dropdown(
                                    id='pair-dropdown',
                                    options=pair_options,
                                    value=pair_options[0]['value'] if pair_options else None,
                                    placeholder="Choose a cointegrated pair...",
                                    style={'backgroundColor': colors['secondary'], 'color': 'black'}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Analysis Start Date", style={'color': colors['text'], 'fontWeight': 'bold'}),
                                dcc.DatePickerSingle(
                                    id='analysis-start-date',
                                    date=pairs_data['start_date'],
                                    style={'width': '100%'}
                                ),
                                html.Small("Different from screening period", style={'color': colors['accent']})
                            ], width=3),
                            dbc.Col([
                                html.Label("Analysis End Date", style={'color': colors['text'], 'fontWeight': 'bold'}),
                                dcc.DatePickerSingle(
                                    id='analysis-end-date',
                                    date=pairs_data['end_date'],
                                    style={'width': '100%'}
                                ),
                                html.Small("For detailed analysis", style={'color': colors['accent']})
                            ], width=3),
                            dbc.Col([
                                dbc.Button("🔄 Update Analysis", id="update-pair-btn", color="info", size="sm",
                                          style={'marginTop': '25px', 'width': '100%'})
                            ], width=2)
                        ]),
                        html.Hr(style={'borderColor': colors['border'], 'margin': '20px 0'}),
                        # Period Information
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.P([
                                        html.Strong("🔍 Screening Period: ", style={'color': colors['accent']}),
                                        f"{pairs_data['start_date']} to {pairs_data['end_date']}"
                                    ], style={'color': colors['text'], 'margin': 0}),
                                    html.P([
                                        html.Strong("📊 Analysis Period: ", style={'color': colors['info']}),
                                        html.Span(id="current-analysis-period", children=f"{pairs_data['start_date']} to {pairs_data['end_date']}")
                                    ], style={'color': colors['text'], 'margin': 0})
                                ])
                            ], width=12)
                        ])
                    ], style=card_style)
                ], width=12)
            ]),
            
            # Analysis Results
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("📈 Pair Statistics", style={'color': colors['text']}),
                        html.Div(id="pair-stats")
                    ], style=card_style)
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H4("🎯 Stock1 × Stock2 Relationship", style={'color': colors['text']}),
                        dcc.Graph(id="2d-plot", style={'height': '400px'})
                    ], style=card_style)
                ], width=8)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("📊 Spread Analysis", style={'color': colors['text']}),
                        dcc.Graph(id="spread-plot", style={'height': '400px'})
                    ], style=card_style)
                ], width=12)
            ])
        ])

# Enhanced callback for pair analysis with separate analysis dates
@callback(
    [Output('pair-stats', 'children'),
     Output('2d-plot', 'figure'),
     Output('spread-plot', 'figure'),
     Output('current-analysis-period', 'children')],
    [Input('pair-dropdown', 'value'),
     Input('update-pair-btn', 'n_clicks')],
    [dash.State('pairs-store', 'data'),
     dash.State('analysis-start-date', 'date'),
     dash.State('analysis-end-date', 'date')]
)
def update_pair_analysis(selected_pair, update_clicks, pairs_data, analysis_start, analysis_end):
    empty_fig = create_styled_figure()
    period_text = f"{pairs_data['start_date']} to {pairs_data['end_date']}" if pairs_data else "No data"
    
    if not selected_pair or not pairs_data:
        return html.P("🔍 Select a pair to analyze", style={'color': colors['text'], 'textAlign': 'center'}), empty_fig, empty_fig, period_text
    
    # Use analysis dates if provided, otherwise fall back to screening dates
    start_date = analysis_start if analysis_start else pairs_data['start_date']
    end_date = analysis_end if analysis_end else pairs_data['end_date']
    
    # Update period display
    period_text = f"{start_date} to {end_date}"
    
    # Parse selected assets (can be pair or triplet)
    assets = selected_pair.split('|')
    
    # Get data for the analysis period
    pair_data = get_pair_data(assets, start_date, end_date)
    
    if len(pair_data) < 2:
        return html.P("❌ Error loading asset data for analysis period", style={'color': colors['warning']}), empty_fig, empty_fig, period_text
    
    # For display purposes, use first two assets for spread analysis
    main_assets = assets[:2]
    main_pair_data = {asset: pair_data[asset] for asset in main_assets if asset in pair_data}
    
    # Calculate spread and statistics for main pair
    stats = calculate_spread_and_stats(main_pair_data)
    
    if not stats:
        return html.P("❌ Error calculating statistics for analysis period", style={'color': colors['warning']}), empty_fig, empty_fig, period_text
    
    # Recalculate p-value for the analysis period
    try:
        cointegration_result = multi_cointegration_test(pair_data)
        p_val = cointegration_result['p_value']
        if len(assets) > 2:
            test_type = "Johansen (multivariate)"
        else:
            test_type = "Phillips-Ouliaris (pairwise)"
            
        # Additional statistics from cointegration test
        if 'critical_values' in cointegration_result:
            crit_vals = cointegration_result['critical_values']
            is_significant_1pct = p_val < 0.01
            is_significant_5pct = p_val < 0.05
            is_significant_10pct = p_val < 0.10
        else:
            crit_vals = None
            is_significant_1pct = is_significant_5pct = is_significant_10pct = False
            
    except Exception as e:
        p_val = "N/A"
        test_type = "Error"
        crit_vals = None
        is_significant_1pct = is_significant_5pct = is_significant_10pct = False
        print(f"Error in cointegration test: {e}")
    
    # Enhanced statistics display
    stock1, stock2 = main_assets
    r_squared = stats['lr_full'].score(stats['aligned_data'][stock2].values.reshape(-1, 1), stats['aligned_data'][stock1].values)
    
    stats_div = [
        html.Div([
            html.H5(f"📊 {' ↔ '.join(assets)}", style={'color': colors['accent'], 'marginBottom': '15px'}),
            
            # Cointegration Test Results
            html.Div([
                html.H6("🧪 Cointegration Test", style={'color': colors['info'], 'marginBottom': '10px'}),
                html.P([html.Strong("Test Type: "), test_type], style={'color': colors['text'], 'marginBottom': '5px'}),
                html.P([
                    html.Strong("P-value: "), 
                    html.Span(f"{p_val:.6f}" if isinstance(p_val, float) else f"{p_val}", 
                             style={'color': colors['success'] if isinstance(p_val, float) and p_val < 0.05 else colors['warning']})
                ], style={'marginBottom': '5px'}),
                
                # Significance levels
                html.Div([
                    html.P([
                        html.Span("✅" if is_significant_1pct else "❌", style={'marginRight': '5px'}),
                        "Significant at 1% level"
                    ], style={'color': colors['success'] if is_significant_1pct else colors['text'], 'marginBottom': '3px', 'fontSize': '0.9rem'}),
                    html.P([
                        html.Span("✅" if is_significant_5pct else "❌", style={'marginRight': '5px'}),
                        "Significant at 5% level"
                    ], style={'color': colors['success'] if is_significant_5pct else colors['text'], 'marginBottom': '3px', 'fontSize': '0.9rem'}),
                    html.P([
                        html.Span("✅" if is_significant_10pct else "❌", style={'marginRight': '5px'}),
                        "Significant at 10% level"
                    ], style={'color': colors['success'] if is_significant_10pct else colors['text'], 'marginBottom': '10px', 'fontSize': '0.9rem'})
                ])
            ], style={'marginBottom': '15px', 'padding': '10px', 'backgroundColor': 'rgba(64,224,208,0.1)', 'borderRadius': '5px'}),
            
            # Regression Statistics
            html.Div([
                html.H6("📈 Regression Statistics", style={'color': colors['info'], 'marginBottom': '10px'}),
                html.P([html.Strong("Correlation: "), f"{stats['correlation']:.4f}"], style={'color': colors['text'], 'marginBottom': '5px'}),
                html.P([html.Strong("R²: "), f"{r_squared:.4f}"], style={'color': colors['text'], 'marginBottom': '5px'}),
                html.P([html.Strong("Beta (spread): "), f"{stats['beta_spread']:.4f}"], style={'color': colors['text'], 'marginBottom': '5px'}),
                html.P([html.Strong("Full period β: "), f"{stats['lr_full'].coef_[0]:.4f}"], style={'color': colors['text']})
            ], style={'marginBottom': '15px', 'padding': '10px', 'backgroundColor': 'rgba(255,165,0,0.1)', 'borderRadius': '5px'}),
            
            # Spread Statistics
            html.Div([
                html.H6("📊 Spread Statistics", style={'color': colors['info'], 'marginBottom': '10px'}),
                html.P([html.Strong("Mean: "), f"{stats['spread'].mean():.4f}"], style={'color': colors['text'], 'marginBottom': '5px'}),
                html.P([html.Strong("Std Dev: "), f"{stats['spread'].std():.4f}"], style={'color': colors['text'], 'marginBottom': '5px'}),
                html.P([html.Strong("Min: "), f"{stats['spread'].min():.4f}"], style={'color': colors['text'], 'marginBottom': '5px'}),
                html.P([html.Strong("Max: "), f"{stats['spread'].max():.4f}"], style={'color': colors['text']})
            ], style={'padding': '10px', 'backgroundColor': 'rgba(0,255,127,0.1)', 'borderRadius': '5px'})
        ])
    ]
    
    if len(assets) > 2:
        stats_div.append(
            html.Div([
                html.P(f"📈 Spread analysis: {stock1} - {stock2}", 
                      style={'color': colors['info'], 'fontStyle': 'italic', 'marginTop': '15px'})
            ])
        )
    
    # Enhanced 2D scatter plot
    scatter_fig = create_styled_figure()
    
    scatter_fig.add_trace(go.Scatter(
        x=stats['aligned_data'][stock2],
        y=stats['aligned_data'][stock1],
        mode='markers',
        marker=dict(
            color=colors['accent'], 
            size=6,
            opacity=0.7,
            line=dict(width=0.5, color='white')
        ),
        name='Price Points',
        hovertemplate=f'<b>{stock2}</b>: $%{{x:.2f}}<br><b>{stock1}</b>: $%{{y:.2f}}<extra></extra>'
    ))
    
    # Add regression line
    x_range = np.linspace(stats['aligned_data'][stock2].min(), stats['aligned_data'][stock2].max(), 100)
    y_pred = stats['lr_full'].predict(x_range.reshape(-1, 1))
    
    scatter_fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        line=dict(color=colors['success'], width=3),
        name='Regression Line',
        hovertemplate='<b>Regression</b><br>%{x:.2f} → %{y:.2f}<extra></extra>'
    ))
    
    scatter_fig.update_layout(
        title=dict(
            text=f"{stock1} vs {stock2} • R² = {r_squared:.3f} • p = {p_val:.4f}" if isinstance(p_val, float) else f"{stock1} vs {stock2} • R² = {r_squared:.3f}",
            x=0.5
        ),
        xaxis_title=f"{stock2} Price ($)",
        yaxis_title=f"{stock1} Price ($)",
        height=400
    )
    
    # Enhanced spread plot
    spread_fig = create_styled_figure()
    
    spread_fig.add_trace(go.Scatter(
        x=stats['spread'].index,
        y=stats['spread'],
        mode='lines',
        line=dict(color=colors['text'], width=2),
        name='Spread',
        hovertemplate='<b>%{x}</b><br>Spread: %{y:.2f}<extra></extra>'
    ))
    
    # Add statistical bands
    spread_mean = stats['spread'].mean()
    spread_std = stats['spread'].std()
    
    spread_fig.add_hline(
        y=spread_mean, 
        line_dash="dash", 
        line_color=colors['accent'], 
        annotation_text=f"Mean: {spread_mean:.2f}",
        annotation_position="bottom right"
    )
    
    spread_fig.add_hline(
        y=spread_mean + 2*spread_std, 
        line_dash="dot", 
        line_color=colors['warning'], 
        annotation_text=f"+2σ: {spread_mean + 2*spread_std:.2f}",
        annotation_position="top right"
    )
    
    spread_fig.add_hline(
        y=spread_mean - 2*spread_std, 
        line_dash="dot", 
        line_color=colors['warning'], 
        annotation_text=f"-2σ: {spread_mean - 2*spread_std:.2f}",
        annotation_position="bottom right"
    )
    
    # Add background shading for trading zones
    spread_fig.add_hrect(
        y0=spread_mean + 2*spread_std, 
        y1=spread_mean + 3*spread_std, 
        fillcolor="rgba(255,99,71,0.2)", 
        layer="below", 
        line_width=0
    )
    
    spread_fig.add_hrect(
        y0=spread_mean - 2*spread_std, 
        y1=spread_mean - 3*spread_std, 
        fillcolor="rgba(255,99,71,0.2)", 
        layer="below", 
        line_width=0
    )
    
    # Add period information to spread plot title
    spread_fig.update_layout(
        title=dict(
            text=f"Spread: {stock1} - {stats['beta_spread']:.4f} × {stock2} | Period: {start_date} to {end_date}",
            x=0.5
        ),
        xaxis_title="Date",
        yaxis_title="Spread Value",
        height=400
    )
    
    return stats_div, scatter_fig, spread_fig, period_text

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050) 