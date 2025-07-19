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
    if not os.path.exists(DIRECTORY):
        print(f"Error: Directory {DIRECTORY} not found.")
        return {}
    
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]
    vector_universe = {} 
    
    for file_name in files:
        try:
            df = pd.read_csv(os.path.join(DIRECTORY, file_name), index_col=0)
            df = df[start_date:end_date]
            df_price = df['Close']
            
            if len(df_price) < LOOKBACK * 3:
                print(f'COULD NOT EVAL {file_name}: {len(df_price)}') 
                continue
                
            print(f"Processing {file_name} {len(df_price)}") 
            
            # Rolling z-score (same as original)
            rolling_mean = df_price.rolling(window=LOOKBACK).mean()
            rolling_std = df_price.rolling(window=LOOKBACK).std()
            rolling_zscore = (df_price - rolling_mean) / rolling_std
            rolling_zscore = rolling_zscore.dropna(how='all').to_list()
            
            x = stride_window(STRIDE, rolling_zscore, LOOKBACK)
            
            # Encode using the loaded encoder
            stock_name = ''.join([letter for letter in file_name if letter.isupper()])
            stock_vect = encoder.predict(x).squeeze()
            vector_universe[stock_name] = stock_vect.tolist()
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    
    return vector_universe

def cluster_and_test_pairs(vector_universe, start_date, end_date, alpha=0.1, include_triplets=False):
    """
    Cluster encoded vectors and test cointegration for pairs (and optionally triplets)
    Returns: (pairs_with_pvals, cluster_info)
    """
    if not vector_universe:
        return [], {}
    
    cluster_model = HDBSCAN(min_cluster_size=6)
    data_labels = list(vector_universe.keys())
    vector_data = list(vector_universe.values())
    
    # Flatten and pad vectors (same as original)
    flatten_to_2d = lambda data: [[i for sub in ([el] if not isinstance(el, list) else el) for i in (sub if not isinstance(sub, list) else [x for x in sub])] if not isinstance(item, list) else [i for sub in item for i in (sub if not isinstance(sub, list) else [x for x in sub])] for item in data]
    pad = lambda lst: [x + [0] * (max(map(len, lst)) - len(x)) for x in lst]
    vector_data = pad(flatten_to_2d(vector_data))
    
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
    
    # Print cluster information before cointegration testing
    print("\n" + "="*60)
    print("🔍 CLUSTER ANALYSIS RESULTS")
    print("="*60)
    
    total_clustered = 0
    valid_clusters = 0
    
    for cluster_id in sorted(sorted_clusters.keys()):
        stocks = sorted_clusters[cluster_id]
        if cluster_id == -1:
            print(f"🔸 Noise (Cluster {cluster_id}): {len(stocks)} stocks")
            print(f"   Stocks: {', '.join(stocks[:10])}{'...' if len(stocks) > 10 else ''}")
        else:
            print(f"📊 Cluster {cluster_id}: {len(stocks)} stocks")
            print(f"   Stocks: {', '.join(stocks)}")
            total_clustered += len(stocks)
            valid_clusters += 1
    
    print(f"\n📈 Summary:")
    print(f"   • Total valid clusters: {valid_clusters}")
    print(f"   • Total stocks in clusters: {total_clustered}")
    print(f"   • Noise points: {len(sorted_clusters.get(-1, []))}")
    print(f"   • Total stocks processed: {len(data_labels)}")
    print("="*60 + "\n")
    
    # Generate pairs (and optionally triplets) from clusters
    for key, lst in sorted_clusters.items():
        if len(lst) >= 2:
            # Generate pairs
            pairs = list(combinations(lst, 2))
            for item in pairs:
                combo_universe.append(item)
            
            # Generate triplets if requested and cluster is large enough
            if include_triplets and len(lst) >= 3:
                triplets = list(combinations(lst, 3))
                for item in triplets:
                    combo_universe.append(item)
    
    print(f"🧪 Testing {len(combo_universe)} combinations for cointegration...")
    
    # Test cointegration for all combinations
    final_p_vals = {}
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]   
    
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
            
            if len(asset_data) >= 2:  # Support both pairs and triplets
                result = multi_cointegration_test(asset_data)
                if len(asset_data) == 2:
                    p_val = result['p_value']
                else:  # Triplets or more - use Johansen test result
                    # For Johansen test, we'll use the trace statistic comparison
                    # A more sophisticated approach would be needed for production
                    p_val = 0.05 if result['cointegration_rank'] > 0 else 1.0
                final_p_vals[assets] = p_val 
        except Exception as e:
            print(f"Error testing {assets}: {e}")
            continue
    
    top_pairs = [(item[0], item[1]) for item in final_p_vals.items() if item[1] <= alpha]
    
    print(f"✅ Found {len(top_pairs)} cointegrated pairs with p-value ≤ {alpha}")
    
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

# Tab content callback
@callback(
    Output("tab-content", "children"),
    [Input("main-tabs", "value")]
)
def render_tab_content(active_tab):
    if active_tab == "clusters":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("🎯 Cluster Information", style={'color': colors['text']}),
                        html.Div(id="cluster-info")
                    ], style=card_style)
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("🗺️ Cluster Visualization (PCA)", style={'color': colors['text']}),
                        dcc.Graph(id="cluster-plot", style={'height': '500px'})
                    ], style=card_style)
                ], width=12)
            ])
        ])
    
    elif active_tab == "pairs":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("📋 Select Trading Pair", style={'color': colors['text']}),
                        dcc.Dropdown(
                            id='pair-dropdown',
                            placeholder="Choose a cointegrated pair...",
                            style={'backgroundColor': colors['secondary'], 'color': 'black'}
                        )
                    ], style=card_style)
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("📈 Pair Statistics", style={'color': colors['text']}),
                        html.Div(id="pair-stats")
                    ], style=card_style)
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H4("🎯 Price Relationship", style={'color': colors['text']}),
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

# Callbacks
@callback(
    [Output('pair-dropdown', 'options'),
     Output('pair-dropdown', 'value'),
     Output('cluster-info', 'children'),
     Output('cluster-plot', 'figure')],
    [Input('update-btn', 'n_clicks')],
    [dash.State('start-date', 'date'),
     dash.State('end-date', 'date'),
     dash.State('alpha-input', 'value'),
     dash.State('triplets-checkbox', 'value')]
)
def update_analysis(n_clicks, start_date, end_date, alpha, triplets_enabled):
    if n_clicks is None:
        n_clicks = 1  # Initial load
    
    include_triplets = 'enable' in triplets_enabled if triplets_enabled else False
    
    # Encode stock data
    vector_universe = encode_stock_data(encoder, start_date, end_date)
    
    # Cluster and test pairs (and optionally triplets)
    pairs_with_pvals, cluster_info = cluster_and_test_pairs(vector_universe, start_date, end_date, alpha, include_triplets)
    
    # Prepare dropdown options
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
    
    selected_pair = pair_options[0]['value'] if pair_options else None
    
    # Cluster information with enhanced styling
    clusters = cluster_info.get('clusters', {})
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
    
    # Cluster visualization using PCA with enhanced styling
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
    
    return pair_options, selected_pair, cluster_info_div, cluster_fig

@callback(
    [Output('pair-stats', 'children'),
     Output('2d-plot', 'figure'),
     Output('spread-plot', 'figure')],
    [Input('pair-dropdown', 'value')],
    [dash.State('start-date', 'date'),
     dash.State('end-date', 'date')]
)
def update_pair_analysis(selected_pair, start_date, end_date):
    empty_fig = create_styled_figure()
    
    if not selected_pair:
        return html.P("🔍 Select a pair to analyze", style={'color': colors['text'], 'textAlign': 'center'}), empty_fig, empty_fig
    
    # Parse selected assets (can be pair or triplet)
    assets = selected_pair.split('|')
    
    # Get data for all assets
    pair_data = get_pair_data(assets, start_date, end_date)
    
    if len(pair_data) < 2:
        return html.P("❌ Error loading asset data", style={'color': colors['warning']}), empty_fig, empty_fig
    
    # For display purposes, use first two assets for spread analysis
    main_assets = assets[:2]
    main_pair_data = {asset: pair_data[asset] for asset in main_assets if asset in pair_data}
    
    # Calculate spread and statistics for main pair
    stats = calculate_spread_and_stats(main_pair_data)
    
    if not stats:
        return html.P("❌ Error calculating statistics", style={'color': colors['warning']}), empty_fig, empty_fig
    
    # Recalculate p-value for all selected assets
    try:
        p_val = multi_cointegration_test(pair_data)['p_value']
        if len(assets) > 2:
            test_type = "Johansen (multivariate)"
        else:
            test_type = "Phillips-Ouliaris (pairwise)"
    except:
        p_val = "N/A"
        test_type = "Error"
    
    # Enhanced statistics display
    stock1, stock2 = main_assets
    stats_div = [
        html.Div([
            html.H5(f"📊 {' ↔ '.join(assets)}", style={'color': colors['accent'], 'marginBottom': '15px'}),
            html.Div([
                html.P([html.Strong("Test Type: "), test_type], style={'color': colors['text'], 'marginBottom': '8px'}),
                html.P([
                    html.Strong("P-value: "), 
                    html.Span(f"{p_val:.6f}" if isinstance(p_val, float) else f"{p_val}", 
                             style={'color': colors['success'] if isinstance(p_val, float) and p_val < 0.05 else colors['warning']})
                ], style={'marginBottom': '8px'}),
                html.P([html.Strong("Correlation: "), f"{stats['correlation']:.4f}"], style={'color': colors['text'], 'marginBottom': '8px'}),
                html.P([html.Strong("Beta (spread): "), f"{stats['beta_spread']:.4f}"], style={'color': colors['text'], 'marginBottom': '8px'}),
                html.P([html.Strong("Full period β: "), f"{stats['lr_full'].coef_[0]:.4f}"], style={'color': colors['text'], 'marginBottom': '8px'}),
                html.P([html.Strong("R²: "), f"{stats['lr_full'].score(stats['aligned_data'][stock2].values.reshape(-1, 1), stats['aligned_data'][stock1].values):.4f}"], style={'color': colors['text']})
            ])
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
            text=f"{stock1} vs {stock2} • R² = {stats['lr_full'].score(stats['aligned_data'][stock2].values.reshape(-1, 1), stats['aligned_data'][stock1].values):.3f}",
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
    
    spread_fig.update_layout(
        title=dict(
            text=f"Spread: {stock1} - {stats['beta_spread']:.4f} × {stock2}",
            x=0.5
        ),
        xaxis_title="Date",
        yaxis_title="Spread Value",
        height=400
    )
    
    return stats_div, scatter_fig, spread_fig

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050) 