import random 
from datetime import datetime, timedelta
import pandas as pd
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
import numpy as np
from itertools import combinations
from tensorflow.keras.optimizers import AdamW
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Import the functions from the original script
DIRECTORY = '/home/kisero/ML-shyt-public/stock_data/'
LOOKBACK  = 100 # Input into autoencoder

def multi_cointegration_test(
    data_dict,
    det_order=0,
    k_ar_diff=1,
    significance=0.05
):
    """
    Test cointegration among N time series provided as lists, arrays, or pandas Series.
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

def build_model(output_len):
    model = Sequential([
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(.2),
        Dense(16, activation='gelu', kernel_regularizer=l2(0.01)), 
        Dropout(.1), 
        Dense(8,activation='gelu', kernel_regularizer=l2(0.01)),
        Dense(32, activation='relu'), 
        Dense(output_len,activation='relu') # Output layer
        ])
    model.compile(optimizer=AdamW(), loss='mse')
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

def train_autoencoder(LOOKBACK, start, end, alpha, model):
    """
    Given start, end dates screens potential pairs
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
        x = y = stride_window(64, rolling_zscore, LOOKBACK )
        # Random validation sets
        idx1, idx2, idx3 = random.sample(range(0, len(x) - 2), 3) 
        x_val = y_val = np.array([x[idx1], x[idx2], x[idx3]])
        # Subtract validation sets from train
        x = y = [x[sample_idx] for sample_idx in range(len(x)) if sample_idx not in [idx1,idx2,idx3]] 
        x, y = np.array(x), np.array(y) 
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=False) # earlystop function
        model.fit(np.array(x), np.array(y) , epochs=1000, validation_data=(x_val, y_val), callbacks=early_stop) # Fit model with earling stopping callback
        # Encode newest data
        encoder = Sequential()
        for i in range(5): #range(n) where n is layer of latent space
            encoder.add(model.layers[i])
        stock_name = ''.join([letter for letter in file_name if letter.isupper()])
        #return model  
        
        stock_vect = encoder.predict(x).squeeze() # Squeezes and predicts the most re 
        stock_vect.tolist()# Turns numpy into vanilla python lst 
        vector_universe[stock_name] = stock_vect.tolist()
    return vector_universe, encoder        

def create_futuristic_tsne_plot(vector_universe, background_image_path):
    """
    Create an interactive t-SNE plot with futuristic styling and background image
    """
    # Extract stock names and vectors
    stock_names = list(vector_universe.keys())
    vectors = list(vector_universe.values())
    
    # Convert vectors to numpy array for t-SNE
    # If vectors are 2D (multiple vectors per stock), flatten or take mean
    processed_vectors = []
    for vec in vectors:
        if isinstance(vec, list) and len(vec) > 0:
            if isinstance(vec[0], list):  # 2D vector
                processed_vectors.append(np.mean(vec, axis=0))  # Take mean
            else:  # 1D vector
                processed_vectors.append(vec)
        else:
            processed_vectors.append(vec)
    
    vectors_array = np.array(processed_vectors)
    
    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors_array)-1))
    tsne_results = tsne.fit_transform(vectors_array)
    
    # Apply clustering for color coding
    print("Applying clustering...")
    cluster_model = HDBSCAN(min_cluster_size=max(2, len(vectors_array)//10))
    cluster_labels = cluster_model.fit_predict(vectors_array)
    
    # Create color palette for clusters
    unique_clusters = np.unique(cluster_labels)
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
    
    # Load and encode background image
    try:
        img = Image.open(background_image_path)
        img_width, img_height = img.size
        
        # Convert image to base64 for plotly
        import io
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
    except Exception as e:
        print(f"Could not load background image: {e}")
        img_base64 = None
        img_width, img_height = 1000, 1000
    
    # Create the plot
    fig = go.Figure()
    
    # Add background image if available
    if img_base64:
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{img_base64}",
                xref="paper", yref="paper",
                x=0, y=1, sizex=1, sizey=1,
                xanchor="left", yanchor="top",
                opacity=0.3,
                layer="below"
            )
        )
    
    # Add scatter plot for each cluster
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_color = colors[cluster_id % len(colors)] if cluster_id != -1 else '#808080'
        cluster_name = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
        
        fig.add_trace(go.Scatter(
            x=tsne_results[mask, 0],
            y=tsne_results[mask, 1],
            mode='markers+text',
            text=[stock_names[i] for i in range(len(stock_names)) if mask[i]],
            textposition="middle center",
            textfont=dict(color='white', size=10, family='Courier New'),
            name=cluster_name,
            marker=dict(
                size=15,
                color=cluster_color,
                line=dict(width=2, color='rgba(0,255,255,0.8)'),
                opacity=0.8
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Cluster: ' + cluster_name + '<br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<extra></extra>'
        ))
    
    # Update layout with futuristic styling
    fig.update_layout(
        title=dict(
            text="<b>Stock Vector Clustering - t-SNE Visualization</b>",
            x=0.5,
            font=dict(size=24, color='cyan', family='Courier New')
        ),
        xaxis=dict(
            title="t-SNE Dimension 1",
            tickfont=dict(color='white', size=12, family='Courier New'),
            gridcolor='rgba(0,255,255,0.2)',
            gridwidth=1,
            zerolinecolor='rgba(0,255,255,0.5)',
            zerolinewidth=2
        ),
        yaxis=dict(
            title="t-SNE Dimension 2",
            tickfont=dict(color='white', size=12, family='Courier New'),
            gridcolor='rgba(0,255,255,0.2)',
            gridwidth=1,
            zerolinecolor='rgba(0,255,255,0.5)',
            zerolinewidth=2
        ),
        plot_bgcolor='rgba(0,0,0,0.9)',
        paper_bgcolor='rgba(0,0,0,0.95)',
        font=dict(color='white', family='Courier New'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='cyan',
            borderwidth=2,
            font=dict(color='white', family='Courier New')
        ),
        hovermode='closest',
        width=1200,
        height=800
    )
    
    # Add some futuristic effects
    fig.update_traces(
        marker=dict(
            line=dict(width=2, color='rgba(0,255,255,0.8)')
        )
    )
    
    return fig

def main():
    """
    Main function to run the autoencoder and create the visualization
    """
    # Parameters
    lookback = 100
    start_date = '2014-01-01'
    end_date = '2020-01-01'
    alpha = 0.1
    
    # Background image path
    background_image_path = '/home/kisero/Downloads/asdf.jpg'
    
    # Build model
    print("Building autoencoder model...")
    model = build_model(lookback)
    
    # Train autoencoder and get vectors
    print("Training autoencoder and extracting vectors...")
    vector_universe, encoder = train_autoencoder(lookback, start_date, end_date, alpha, model)
    
    if not vector_universe:
        print("No vectors generated. Please check your data directory and files.")
        return
    
    print(f"Generated vectors for {len(vector_universe)} stocks")
    
    # Create the futuristic t-SNE visualization
    print("Creating t-SNE visualization...")
    fig = create_futuristic_tsne_plot(vector_universe, background_image_path)
    
    # Save as HTML file
    output_file = 'futuristic_tsne_visualization.html'
    fig.write_html(output_file)
    print(f"Interactive visualization saved as {output_file}")
    
    # Show the plot
    fig.show()

if __name__ == "__main__":
    main() 
