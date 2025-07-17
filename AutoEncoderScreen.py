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

# Configuration
DIRECTORY = 'stock_data/Data/'
LOOKBACK = 256  # Must match training lookback
ENCODER_PATH = "encoder.pkl"

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

def screen_stocks(encoder, start_date, end_date, stride=64, alpha=0.05):
    """
    Screen stocks using the trained encoder and same preprocessing pipeline
    """
    print(f"🔍 Screening stocks from {start_date} to {end_date}")
    print(f"📊 Using alpha = {alpha} for cointegration test")
    
    # Get vector universe using trained encoder
    vector_universe = get_vector_universe(encoder, start_date, end_date, stride)
    
    if not vector_universe:
        print("❌ No valid stock data found!")
        return []
    
    print(f"📈 Found {len(vector_universe)} valid stocks for screening")
    
    # Find cointegrated pairs using clustering
    cointegrated_pairs = cluster_pvals(vector_universe, start_date, end_date, alpha)
    
    if not cointegrated_pairs:
        print(f"❌ No cointegrated pairs found with alpha = {alpha}")
        return []
    
    print(f"✅ Found {len(cointegrated_pairs)} cointegrated pairs")
    
    # Sort by p-value and take top 10
    sorted_pairs = sorted(cointegrated_pairs, key=lambda x: x[1])[:10]
    
    return sorted_pairs

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
            
            if len(df_price) < LOOKBACK + 100:  # Need LOOKBACK for rolling + some extra for processing
                print(f"⚠️  Skipping {file_name}: insufficient data ({len(df_price)} points)")
                continue
                
            print(f"📊 Processing {file_name}: {len(df_price)} data points")
            
            # Apply same preprocessing: rolling z-score
            rolling_mean = df_price.rolling(window=LOOKBACK).mean()
            rolling_std = df_price.rolling(window=LOOKBACK).std()
            rolling_zscore = (df_price - rolling_mean) / rolling_std
            rolling_zscore = rolling_zscore.dropna(how='all').to_list()
            
            # Create windows
            x = stride_window(stride, rolling_zscore, LOOKBACK)
            
            if len(x) == 0:
                continue
                
            # Encode using trained encoder
            stock_vect = encoder.predict(x).squeeze()
            stock_name = ''.join([letter for letter in file_name if letter.isupper()])
            vector_universe[stock_name] = stock_vect.tolist()
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            continue
    
    return vector_universe

def cluster_pvals(vector_universe, start_date, end_date, alpha):
    """
    Cluster stocks and find cointegrated pairs
    """
    print("🔬 Clustering stocks and testing cointegration...")
    
    cluster_model = HDBSCAN(min_cluster_size=6)
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
    combo_universe = []
    
    for key, value in VECTOR_CLUSTER.items():
        if value not in sorted_clusters.keys():
            sorted_clusters[value] = [key]
        else:
            sorted_clusters[value].append(key)
    
    for key, lst in sorted_clusters.items():
        if len(lst) >= 2:
            combos = list(combinations(lst, 2))
            combo_universe.extend(combos)
    
    print(f"🔍 Testing {len(combo_universe)} potential pairs for cointegration...")
    
    # Test cointegration for all pairs
    final_p_vals = {}
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]
    
    for assets in tqdm(combo_universe, desc="Testing cointegration"):
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
            
            if len(asset_data) == 2:
                p_val = multi_cointegration_test(asset_data)['p_value']
                final_p_vals[assets] = p_val
                
        except Exception as e:
            print(f"❌ Error testing {assets}: {e}")
            continue
    
    # Filter by alpha
    top_vals = [(item[0], item[1]) for item in final_p_vals.items() if item[1] <= alpha]
    return top_vals

def calculate_spread_and_ratio(stock1_name, stock2_name, start_date, end_date):
    """
    Calculate spread and ratio for a stock pair
    """
    files = [f for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]
    
    # Load data for both stocks
    stock1_file = [f for f in files if stock1_name in f][0]
    stock2_file = [f for f in files if stock2_name in f][0]
    
    df1 = pd.read_csv(DIRECTORY + stock1_file, index_col=0)
    df2 = pd.read_csv(DIRECTORY + stock2_file, index_col=0)
    
    df1 = df1[start_date:end_date]
    df2 = df2[start_date:end_date]
    
    df1.index = pd.to_datetime(df1.index)
    df2.index = pd.to_datetime(df2.index)
    
    # Align dataframes
    combined_df = pd.concat([df1['Close'], df2['Close']], axis=1, keys=[stock1_name, stock2_name]).dropna()
    
    # Calculate beta using linear regression
    X = combined_df[stock2_name].values.reshape(-1, 1)
    y = combined_df[stock1_name].values
    reg = LinearRegression().fit(X, y)
    beta = reg.coef_[0]
    
    # Calculate spread: stock1 - beta * stock2
    spread = combined_df[stock1_name] - beta * combined_df[stock2_name]
    
    # Calculate ratio: stock1 / stock2
    ratio = combined_df[stock1_name] / combined_df[stock2_name]
    
    return combined_df, spread, ratio, beta

def create_futuristic_dashboard(top_pairs, start_date, end_date):
    """
    Create an interactive futuristic dashboard
    """
    print("🚀 Creating futuristic dashboard...")
    
    # Initialize Dash app with dark theme
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    
    # Custom CSS for futuristic look
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
                    font-family: 'Orbitron', monospace;
                    color: #00ffff;
                }
                .futuristic-container {
                    border: 2px solid #00ffff;
                    border-radius: 15px;
                    background: rgba(0, 255, 255, 0.1);
                    backdrop-filter: blur(10px);
                    box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
                    margin: 10px;
                    padding: 20px;
                }
                .glow-text {
                    text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;
                }
                .dropdown-style {
                    background-color: rgba(0, 0, 0, 0.8) !important;
                    border: 1px solid #00ffff !important;
                    color: #00ffff !important;
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
    pair_options = [
        {'label': f"{pair[0][0]} - {pair[0][1]} (p-val: {pair[1]:.4f})", 'value': f"{pair[0][0]}|{pair[0][1]}"}
        for pair in top_pairs
    ]
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("🚀 AUTOENCODER PAIRS SCREENER 🚀", 
                       className="text-center glow-text mb-4"),
                html.Hr(style={'border-color': '#00ffff'}),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("⚙️ CONTROL PANEL", className="glow-text"),
                    html.Label("Select Cointegrated Pair:", className="glow-text"),
                    dcc.Dropdown(
                        id='pair-selector',
                        options=pair_options,
                        value=pair_options[0]['value'] if pair_options else None,
                        style={'background-color': 'rgba(0,0,0,0.8)', 'border': '1px solid #00ffff', 'color': '#00ffff'}
                    ),
                    html.Br(),
                    html.Label("Display Options:", className="glow-text"),
                    dcc.Checklist(
                        id='display-options',
                        options=[
                            {'label': ' Show Spread', 'value': 'spread'},
                            {'label': ' Show Ratio', 'value': 'ratio'},
                            {'label': ' Show Individual Stocks', 'value': 'stocks'}
                        ],
                        value=['spread', 'ratio'],
                        style={'color': '#00ffff'}
                    ),
                    html.Br(),
                    html.Label("Background Style:", className="glow-text"),
                    dcc.Dropdown(
                        id='background-selector',
                        options=[
                            {'label': 'Dark Matrix', 'value': 'dark'},
                            {'label': 'Neon Grid', 'value': 'neon'},
                            {'label': 'Space Blue', 'value': 'space'},
                            {'label': 'Cyber Purple', 'value': 'cyber'}
                        ],
                        value='dark',
                        style={'background-color': 'rgba(0,0,0,0.8)', 'border': '1px solid #00ffff', 'color': '#00ffff'}
                    )
                ], className="futuristic-container")
            ], width=3),
            
            dbc.Col([
                html.Div([
                    dcc.Graph(id='main-chart', style={'height': '600px'})
                ], className="futuristic-container")
            ], width=9)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("📊 PAIR STATISTICS", className="glow-text"),
                    html.Div(id='stats-display')
                ], className="futuristic-container")
            ], width=12)
        ])
    ], fluid=True)
    
    @app.callback(
        [Output('main-chart', 'figure'),
         Output('stats-display', 'children')],
        [Input('pair-selector', 'value'),
         Input('display-options', 'value'),
         Input('background-selector', 'value')]
    )
    def update_dashboard(selected_pair, display_options, background_style):
        if not selected_pair:
            return {}, "No pair selected"
        
        stock1, stock2 = selected_pair.split('|')
        
        try:
            # Calculate spread and ratio
            combined_df, spread, ratio, beta = calculate_spread_and_ratio(stock1, stock2, start_date, end_date)
            
            # Background styles
            bg_styles = {
                'dark': {'plot_bgcolor': '#0c0c0c', 'paper_bgcolor': 'rgba(0,0,0,0)'},
                'neon': {'plot_bgcolor': '#1a1a2e', 'paper_bgcolor': 'rgba(26,26,46,0.8)'},
                'space': {'plot_bgcolor': '#16213e', 'paper_bgcolor': 'rgba(22,33,62,0.8)'},
                'cyber': {'plot_bgcolor': '#2d1b69', 'paper_bgcolor': 'rgba(45,27,105,0.8)'}
            }
            
            # Create subplots
            subplot_count = len([opt for opt in display_options if opt in ['spread', 'ratio', 'stocks']])
            if subplot_count == 0:
                subplot_count = 1
                
            fig = make_subplots(
                rows=subplot_count, cols=1,
                subplot_titles=[],
                vertical_spacing=0.08
            )
            
            current_row = 1
            
            # Add traces based on display options
            if 'stocks' in display_options:
                fig.add_trace(
                    go.Scatter(x=combined_df.index, y=combined_df[stock1], 
                              name=f'{stock1}', line=dict(color='#00ffff', width=2)),
                    row=current_row, col=1
                )
                fig.add_trace(
                    go.Scatter(x=combined_df.index, y=combined_df[stock2], 
                              name=f'{stock2}', line=dict(color='#ff00ff', width=2)),
                    row=current_row, col=1
                )
                current_row += 1
            
            if 'spread' in display_options:
                fig.add_trace(
                    go.Scatter(x=spread.index, y=spread, 
                              name=f'Spread ({stock1} - {beta:.3f}*{stock2})', 
                              line=dict(color='#ffff00', width=2)),
                    row=current_row, col=1
                )
                # Add zero line for spread
                fig.add_hline(y=0, line_dash="dash", line_color="#ffffff", 
                             row=current_row, col=1, opacity=0.5)
                current_row += 1
            
            if 'ratio' in display_options:
                fig.add_trace(
                    go.Scatter(x=ratio.index, y=ratio, 
                              name=f'Ratio ({stock1}/{stock2})', 
                              line=dict(color='#00ff00', width=2)),
                    row=current_row, col=1
                )
                # Add mean line for ratio
                mean_ratio = ratio.mean()
                fig.add_hline(y=mean_ratio, line_dash="dash", line_color="#ffffff", 
                             row=current_row, col=1, opacity=0.5)
            
            # Update layout with futuristic styling
            fig.update_layout(
                title=dict(
                    text=f"<b>🎯 {stock1} vs {stock2} ANALYSIS</b>",
                    font=dict(color='#00ffff', size=20),
                    x=0.5
                ),
                **bg_styles[background_style],
                font=dict(color='#00ffff', family="Orbitron"),
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(0,0,0,0.7)',
                    bordercolor='#00ffff',
                    borderwidth=1
                ),
                xaxis=dict(
                    gridcolor='rgba(0,255,255,0.3)',
                    color='#00ffff',
                    showgrid=True
                ),
                yaxis=dict(
                    gridcolor='rgba(0,255,255,0.3)',
                    color='#00ffff',
                    showgrid=True
                ),
                hovermode='x unified'
            )
            
            # Update all subplot axes
            for i in range(1, subplot_count + 1):
                fig.update_xaxes(
                    gridcolor='rgba(0,255,255,0.3)',
                    color='#00ffff',
                    showgrid=True,
                    row=i, col=1
                )
                fig.update_yaxes(
                    gridcolor='rgba(0,255,255,0.3)',
                    color='#00ffff',
                    showgrid=True,
                    row=i, col=1
                )
            
            # Statistics display
            p_val = next((pair[1] for pair in top_pairs if f"{pair[0][0]}|{pair[0][1]}" == selected_pair), None)
            
            stats = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.P(f"📈 Stock 1: {stock1}", className="glow-text"),
                        html.P(f"📉 Stock 2: {stock2}", className="glow-text"),
                        html.P(f"🔢 Beta: {beta:.4f}", className="glow-text"),
                    ], width=4),
                    dbc.Col([
                        html.P(f"📊 P-Value: {p_val:.6f}", className="glow-text"),
                        html.P(f"📏 Spread Mean: {spread.mean():.4f}", className="glow-text"),
                        html.P(f"📐 Spread Std: {spread.std():.4f}", className="glow-text"),
                    ], width=4),
                    dbc.Col([
                        html.P(f"⚖️ Ratio Mean: {ratio.mean():.4f}", className="glow-text"),
                        html.P(f"📊 Ratio Std: {ratio.std():.4f}", className="glow-text"),
                        html.P(f"📅 Data Points: {len(combined_df)}", className="glow-text"),
                    ], width=4)
                ])
            ])
            
            return fig, stats
            
        except Exception as e:
            error_fig = go.Figure()
            error_fig.update_layout(
                title=f"Error: {str(e)}",
                plot_bgcolor='#0c0c0c',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return error_fig, f"Error processing pair: {str(e)}"
    
    return app

def main():
    """
    Main screening application
    """
    print("🚀 AUTOENCODER PAIRS SCREENER INITIATED")
    print("=" * 50)
    
    # Load trained encoder
    encoder = load_encoder()
    if encoder is None:
        return
    
    # Configuration - Using smaller date range to work with available data
    start_date = '2020-01-01'  # Smaller window to ensure sufficient data
    end_date = '2022-01-01'    # 2 years of data should be enough
    alpha = 0.05
    
    print(f"⏰ Screening period: {start_date} to {end_date}")
    print(f"🎯 Alpha threshold: {alpha}")
    
    # Screen for cointegrated pairs
    top_pairs = screen_stocks(encoder, start_date, end_date, alpha=alpha)
    
    if not top_pairs:
        print("❌ No cointegrated pairs found. Try increasing alpha or checking data availability.")
        return
    
    print("\n🏆 TOP COINTEGRATED PAIRS:")
    print("-" * 40)
    for i, (pair, p_val) in enumerate(top_pairs[:10], 1):
        print(f"{i:2d}. {pair[0]} - {pair[1]}: p-value = {p_val:.6f}")
    
    # Create and run dashboard
    app = create_futuristic_dashboard(top_pairs[:10], start_date, end_date)
    
    print("\n🚀 Launching futuristic dashboard...")
    print("📡 Access at: http://127.0.0.1:8050")
    print("🛑 Press Ctrl+C to stop")
    
    app.run_server(debug=True, host='127.0.0.1', port=8050)

if __name__ == "__main__":
    main() 