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

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load encoder and initial data
encoder = load_encoder()
if encoder is None:
    print("Cannot proceed without encoder. Exiting.")
    exit(1)

# Define color scheme
colors = {
    'background': BACKGROUND_COLOR,
    'text': '#FFD700',  # Gold/Yellow
    'secondary': '#333333',
    'accent': '#FFA500',  # Orange
    'success': '#00FF00',  # Green
    'warning': '#FF6600'   # Red-Orange
}

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Autoencoder Pairs Trading Dashboard", 
                   style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': 30})
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Start Date:", style={'color': colors['text']}),
            dcc.DatePickerSingle(
                id='start-date',
                date='2014-01-01',
                style={'width': '100%'}
            )
        ], width=3),
        dbc.Col([
            html.Label("End Date:", style={'color': colors['text']}),
            dcc.DatePickerSingle(
                id='end-date',
                date='2020-01-01',
                style={'width': '100%'}
            )
        ], width=2),
        dbc.Col([
            html.Label("Alpha (p-value threshold):", style={'color': colors['text']}),
            dcc.Input(
                id='alpha-input',
                type='number',
                value=0.1,
                min=0.01,
                max=1.0,
                step=0.01,
                style={'width': '100%'}
            )
        ], width=2),
        dbc.Col([
            html.Label("Include Triplets:", style={'color': colors['text']}),
            dcc.Checklist(
                id='triplets-checkbox',
                options=[{'label': ' Enable', 'value': 'enable'}],
                value=[],
                style={'color': colors['text'], 'marginTop': '5px'}
            )
        ], width=2),
        dbc.Col([
            dbc.Button("Update Analysis", id="update-btn", color="warning", 
                      style={'marginTop': 25, 'width': '100%'})
        ], width=3)
    ], style={'marginBottom': 30}),
    
    dbc.Row([
        dbc.Col([
            html.H3("Cluster Information", style={'color': colors['text']}),
            html.Div(id="cluster-info")
        ], width=12)
    ], style={'marginBottom': 20}),
    
    dbc.Row([
        dbc.Col([
            html.H3("Cluster Visualization", style={'color': colors['text']}),
            dcc.Graph(id="cluster-plot")
        ], width=12)
    ], style={'marginBottom': 30}),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Pair:", style={'color': colors['text']}),
            dcc.Dropdown(
                id='pair-dropdown',
                style={'color': 'black'}
            )
        ], width=12)
    ], style={'marginBottom': 20}),
    
    dbc.Row([
        dbc.Col([
            html.H3("Pair Statistics", style={'color': colors['text']}),
            html.Div(id="pair-stats")
        ], width=6),
        dbc.Col([
            html.H3("2D Stock Relationship", style={'color': colors['text']}),
            dcc.Graph(id="2d-plot")
        ], width=6)
    ], style={'marginBottom': 30}),
    
    dbc.Row([
        dbc.Col([
            html.H3("Spread Analysis", style={'color': colors['text']}),
            dcc.Graph(id="spread-plot")
        ], width=12)
    ])
    
], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})

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
            label = f"{assets[0]} - {assets[1]} (p={p_val:.4f})"
            value = f"{assets[0]}|{assets[1]}"
        else:  # Triplets
            label = f"{' - '.join(assets)} (p={p_val:.4f})"
            value = "|".join(assets)
        pair_options.append({'label': label, 'value': value})
    
    selected_pair = pair_options[0]['value'] if pair_options else None
    
    # Cluster information
    clusters = cluster_info.get('clusters', {})
    cluster_info_div = []
    
    for cluster_id, stocks in clusters.items():
        if cluster_id != -1:  # Exclude noise points
            cluster_info_div.append(
                html.Div([
                    html.H5(f"Cluster {cluster_id}: {len(stocks)} stocks", 
                           style={'color': colors['accent']}),
                    html.P(", ".join(stocks), style={'color': colors['text']})
                ], style={'marginBottom': 15})
            )
    
    # Cluster visualization using PCA
    cluster_fig = go.Figure()
    
    if cluster_info.get('vector_data'):
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        vector_2d = pca.fit_transform(cluster_info['vector_data'])
        
        # Color by cluster
        unique_clusters = set(cluster_info['cluster_assignments'])
        colors_list = px.colors.qualitative.Set1
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = [c == cluster_id for c in cluster_info['cluster_assignments']]
            cluster_data = vector_2d[np.array(mask)]
            cluster_labels = [cluster_info['labels'][j] for j, m in enumerate(mask) if m]
            
            color = colors_list[i % len(colors_list)] if cluster_id != -1 else 'grey'
            name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
            
            cluster_fig.add_trace(go.Scatter(
                x=cluster_data[:, 0],
                y=cluster_data[:, 1],
                mode='markers+text',
                text=cluster_labels,
                textposition='top center',
                marker=dict(color=color, size=10),
                name=name
            ))
    
    cluster_fig.update_layout(
        title="Stock Clusters (PCA Visualization)",
        xaxis_title="First Principal Component",
        yaxis_title="Second Principal Component",
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['secondary'],
        font=dict(color=colors['text']),
        title_font_color=colors['text']
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
    if not selected_pair:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['secondary'],
            font=dict(color=colors['text'])
        )
        return html.P("No pair selected", style={'color': colors['text']}), empty_fig, empty_fig
    
    # Parse selected assets (can be pair or triplet)
    assets = selected_pair.split('|')
    
    # Get data for all assets
    pair_data = get_pair_data(assets, start_date, end_date)
    
    if len(pair_data) < 2:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['secondary'],
            font=dict(color=colors['text'])
        )
        return html.P("Error loading asset data", style={'color': colors['warning']}), empty_fig, empty_fig
    
    # For display purposes, use first two assets for spread analysis
    main_assets = assets[:2]
    main_pair_data = {asset: pair_data[asset] for asset in main_assets if asset in pair_data}
    
    # Calculate spread and statistics for main pair
    stats = calculate_spread_and_stats(main_pair_data)
    
    if not stats:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['secondary'],
            font=dict(color=colors['text'])
        )
        return html.P("Error calculating statistics", style={'color': colors['warning']}), empty_fig, empty_fig
    
    # Recalculate p-value for all selected assets
    try:
        p_val = multi_cointegration_test(pair_data)['p_value']
        if len(assets) > 2:
            # For triplets, show additional info
            test_type = "Johansen (multivariate)"
        else:
            test_type = "Phillips-Ouliaris (pairwise)"
    except:
        p_val = "N/A"
        test_type = "Error"
    
    # Asset statistics
    stock1, stock2 = main_assets
    stats_div = [
        html.H5(f"{' - '.join(assets)}" + (" (Triplet)" if len(assets) > 2 else ""), 
               style={'color': colors['accent']}),
        html.P(f"Test Type: {test_type}", style={'color': colors['text']}),
        html.P(f"P-value: {p_val:.6f}" if isinstance(p_val, float) else f"P-value: {p_val}", 
               style={'color': colors['text']}),
    ]
    
    if len(assets) > 2:
        stats_div.append(html.P(f"Showing spread analysis for: {stock1} - {stock2}", 
                               style={'color': colors['accent'], 'fontStyle': 'italic'}))
    
    stats_div.extend([
        html.P(f"Correlation ({stock1}-{stock2}): {stats['correlation']:.4f}", style={'color': colors['text']}),
        html.P(f"Beta (for spread): {stats['beta_spread']:.4f}", style={'color': colors['text']}),
        html.P(f"Full period beta: {stats['lr_full'].coef_[0]:.4f}", style={'color': colors['text']}),
        html.P(f"Full period R²: {stats['lr_full'].score(stats['aligned_data'][stock2].values.reshape(-1, 1), stats['aligned_data'][stock1].values):.4f}", 
               style={'color': colors['text']})
    ])
    
    # 2D scatter plot
    scatter_fig = go.Figure()
    
    scatter_fig.add_trace(go.Scatter(
        x=stats['aligned_data'][stock2],
        y=stats['aligned_data'][stock1],
        mode='markers',
        marker=dict(color=colors['accent'], size=4),
        name='Data Points'
    ))
    
    # Add regression line
    x_range = np.linspace(stats['aligned_data'][stock2].min(), stats['aligned_data'][stock2].max(), 100)
    y_pred = stats['lr_full'].predict(x_range.reshape(-1, 1))
    
    scatter_fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        line=dict(color=colors['success'], width=2),
        name='Regression Line'
    ))
    
    scatter_fig.update_layout(
        title=f"{stock1} vs {stock2}",
        xaxis_title=stock2,
        yaxis_title=stock1,
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['secondary'],
        font=dict(color=colors['text']),
        title_font_color=colors['text']
    )
    
    # Spread plot
    spread_fig = go.Figure()
    
    spread_fig.add_trace(go.Scatter(
        x=stats['spread'].index,
        y=stats['spread'],
        mode='lines',
        line=dict(color=colors['text']),
        name='Spread'
    ))
    
    # Add mean line
    spread_mean = stats['spread'].mean()
    spread_fig.add_hline(y=spread_mean, line_dash="dash", 
                        line_color=colors['accent'], 
                        annotation_text=f"Mean: {spread_mean:.2f}")
    
    # Add +/- 2 standard deviation lines
    spread_std = stats['spread'].std()
    spread_fig.add_hline(y=spread_mean + 2*spread_std, line_dash="dot", 
                        line_color=colors['warning'], 
                        annotation_text=f"+2σ: {spread_mean + 2*spread_std:.2f}")
    spread_fig.add_hline(y=spread_mean - 2*spread_std, line_dash="dot", 
                        line_color=colors['warning'], 
                        annotation_text=f"-2σ: {spread_mean - 2*spread_std:.2f}")
    
    spread_fig.update_layout(
        title=f"Spread: {stock1} - {stats['beta_spread']:.4f} * {stock2}",
        xaxis_title="Date",
        yaxis_title="Spread",
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['secondary'],
        font=dict(color=colors['text']),
        title_font_color=colors['text']
    )
    
    return stats_div, scatter_fig, spread_fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050) 