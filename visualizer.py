#!/usr/bin/env python3
"""
🚀 ADVANCED DIMENSIONALITY REDUCTION VISUALIZATION SYSTEM
========================================================

State-of-the-art machine learning dimensionality reduction techniques
for visualizing autoencoder latent space representations with futuristic
interactive graphs (2D and 3D) and optional background images.

Features:
- Multiple advanced dimensionality reduction algorithms (t-SNE, PCA, Isomap, MDS)
- Interactive futuristic 2D and 3D visualizations with customizable backgrounds
- Real-time clustering analysis with HDBSCAN
- Export capabilities for further analysis
- Imports autoencoder pipeline from AutoEncoderPairScreenerV2.py

Date: 2025
"""

import random
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
DIRECTORY = './stock_data/'
LOOKBACK = 256
BACKGROUND_IMAGE_PATH = None

FUTURISTIC_COLORS = {
    'primary': '#00FFFF',    # Cyan
    'secondary': '#00FF00',  # Green
    'accent': '#FF00FF',     # Magenta
    'warning': '#FFFF00',    # Yellow
    'background': '#0A0A0A', # Dark
    'text': '#FFFFFF'        # White
}

print("🔧 Using standalone autoencoder implementation...")

# Autoencoder implementation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def build_model(output_len):
    """Build optimized autoencoder model"""
    model = Sequential([
        Dense(224, activation='tanh', kernel_regularizer=l1_l2(l1=0.0001461896279370495, l2=0.0028016351587162596)),
        BatchNormalization(),
        Dropout(0.14881529393791154),
        Dense(int(224 * 0.75)),
        LeakyReLU(),
        Dense(12, activation='gelu'),
        Dense(284, activation='gelu'),
        BatchNormalization(),
        Dropout(0.2123738038749523),
        Dense(output_len, activation='gelu')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.018274508859816022),
        loss=Huber()
    )
    return model

def stride_window(stride, lst, window):
    """Create windowed data with stride"""
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

def train_autoencoder(lookback, start_date, end_date, alpha, model):
    """Train autoencoder and return vectors"""
    files = [f for f in os.listdir(DIRECTORY) if f.endswith('.csv')]
    vector_universe = {}
    
    for file_name in files:
        try:
            df = pd.read_csv(os.path.join(DIRECTORY, file_name), index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df[start_date:end_date]
            df_price = df['Close']
            
            if len(df_price) < lookback * 2:
                continue
            
            # Rolling z-score normalization
            rolling_mean = df_price.rolling(window=lookback).mean()
            rolling_std = df_price.rolling(window=lookback).std()
            rolling_zscore = (df_price - rolling_mean) / rolling_std
            rolling_zscore = rolling_zscore.dropna().values
            
            # Create windowed data
            x = y = stride_window(64, rolling_zscore, lookback)
            
            if len(x) < 4:
                continue
            
            # Random validation sets
            idx1, idx2, idx3 = random.sample(range(1, len(x) - 2), 3)
            x_val = y_val = np.array([x[idx1], x[idx2], x[idx3]])
            
            # Training data
            x_train = np.array([x[i] for i in range(len(x)) if i not in [idx1, idx2, idx3]])
            y_train = x_train.copy()
            
            # Train model
            early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
            model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val), 
                     callbacks=[early_stop], verbose=0)
            
            # Create encoder (extract first 5 layers to latent space)
            encoder = Sequential()
            for i in range(6):
                encoder.add(model.layers[i])
            
            # Extract latent vectors
            stock_name = ''.join([letter for letter in file_name if letter.isupper()])
            if not stock_name:
                stock_name = file_name.replace('.csv', '')
            
            stock_vectors = encoder.predict(x_train, verbose=0)
            vector_universe[stock_name] = stock_vectors
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            continue
    
    return vector_universe, encoder

# Machine learning imports
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Visualization imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def safe_json_dump(data, filename):
    """Safely dump data to JSON file with proper type conversion"""
    try:
        converted_data = convert_numpy_types(data)
        with open(filename, 'w') as f:
            json.dump(converted_data, f, indent=2)
        print(f"✅ Successfully saved: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving {filename}: {e}")
        return False

# =============================================================================
# AUTOENCODER PIPELINE
# =============================================================================

def extract_autoencoder_vectors_from_pipeline(start_date, end_date, alpha=0.1):
    """Extract latent vectors using the autoencoder pipeline"""
    print("🔥 Extracting autoencoder vectors using pipeline...")
    
    # Build model
    model = build_model(LOOKBACK)
    
    # Use the train_autoencoder function
    try:
        vector_universe, encoder = train_autoencoder(LOOKBACK, start_date, end_date, alpha, model)
        print(f"✅ Successfully extracted vectors for {len(vector_universe)} stocks")
        return vector_universe, encoder
    except Exception as e:
        print(f"❌ Error in autoencoder pipeline: {e}")
        return {}, None

# =============================================================================
# ADVANCED DIMENSIONALITY REDUCTION
# =============================================================================

class AdvancedDimensionalityReducer:
    """State-of-the-art dimensionality reduction with multiple algorithms"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.reducers_2d = {}
        self.reducers_3d = {}
        self.results_2d = {}
        self.results_3d = {}
        
    def prepare_data(self, vector_universe):
        """Prepare data for dimensionality reduction"""
        print("🔧 Preparing data for dimensionality reduction...")
        
        all_vectors = []
        labels = []
        
        for stock_name, vectors in vector_universe.items():
            if isinstance(vectors, list):
                if len(vectors) > 0 and isinstance(vectors[0], list):
                    mean_vector = np.mean(vectors, axis=0)
                    all_vectors.append(mean_vector)
                else:
                    all_vectors.append(vectors)
                labels.append(stock_name)
            elif isinstance(vectors, np.ndarray):
                if len(vectors.shape) == 1:
                    all_vectors.append(vectors)
                else:
                    mean_vector = np.mean(vectors, axis=0)
                    all_vectors.append(mean_vector)
                labels.append(stock_name)
        
        X = np.array(all_vectors)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"📊 Data shape: {X_scaled.shape}")
        print(f"📊 Number of stocks: {len(labels)}")
        
        return X_scaled, labels
    
    def apply_tsne(self, X, n_components=2, perplexity=30, learning_rate=200):
        """Apply t-SNE dimensionality reduction"""
        print(f"🧠 Applying t-SNE - {n_components}D...")
        
        n_samples = X.shape[0]
        if perplexity >= n_samples:
            adjusted_perplexity = max(1, min(n_samples - 1, n_samples // 3))
            print(f"   📊 Adjusting perplexity from {perplexity} to {adjusted_perplexity}")
            perplexity = adjusted_perplexity
        
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=self.random_state,
            max_iter=1000
        )
        
        embedding = reducer.fit_transform(X)
        
        if n_components == 2:
            self.reducers_2d['tsne'] = reducer
            self.results_2d['tsne'] = embedding
        else:
            self.reducers_3d['tsne'] = reducer
            self.results_3d['tsne'] = embedding
        
        return embedding
    
    def apply_pca(self, X, n_components=2):
        """Apply PCA dimensionality reduction"""
        print(f"📊 Applying PCA - {n_components}D...")
        
        reducer = PCA(n_components=n_components, random_state=self.random_state)
        embedding = reducer.fit_transform(X)
        
        if n_components == 2:
            self.reducers_2d['pca'] = reducer
            self.results_2d['pca'] = embedding
        else:
            self.reducers_3d['pca'] = reducer
            self.results_3d['pca'] = embedding
        
        return embedding
    
    def apply_isomap(self, X, n_components=2, n_neighbors=5):
        """Apply Isomap dimensionality reduction"""
        print(f"🗺️  Applying Isomap - {n_components}D...")
        
        n_samples = X.shape[0]
        if n_neighbors >= n_samples:
            n_neighbors = max(2, n_samples - 1)
        
        reducer = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        embedding = reducer.fit_transform(X)
        
        if n_components == 2:
            self.reducers_2d['isomap'] = reducer
            self.results_2d['isomap'] = embedding
        else:
            self.reducers_3d['isomap'] = reducer
            self.results_3d['isomap'] = embedding
        
        return embedding
    
    def apply_mds(self, X, n_components=2):
        """Apply MDS dimensionality reduction"""
        print(f"📐 Applying MDS - {n_components}D...")
        
        reducer = MDS(n_components=n_components, random_state=self.random_state)
        embedding = reducer.fit_transform(X)
        
        if n_components == 2:
            self.reducers_2d['mds'] = reducer
            self.results_2d['mds'] = embedding
        else:
            self.reducers_3d['mds'] = reducer
            self.results_3d['mds'] = embedding
        
        return embedding
    
    def apply_all_methods(self, X):
        """Apply all available dimensionality reduction methods for both 2D and 3D"""
        print("🎯 Applying all dimensionality reduction methods...")
        
        methods = {
            'tsne': self.apply_tsne,
            'pca': self.apply_pca,
            'isomap': self.apply_isomap,
            'mds': self.apply_mds
        }
        
        # Apply 2D methods
        print("\n📊 Applying 2D methods...")
        for method_name, method_func in methods.items():
            try:
                result = method_func(X, n_components=2)
                if result is not None:
                    print(f"✅ {method_name.upper()} 2D completed successfully")
                else:
                    print(f"⚠️  {method_name.upper()} 2D skipped")
            except Exception as e:
                print(f"❌ {method_name.upper()} 2D failed: {e}")
        
        # Apply 3D methods
        print("\n🎲 Applying 3D methods...")
        for method_name, method_func in methods.items():
            try:
                result = method_func(X, n_components=3)
                if result is not None:
                    print(f"✅ {method_name.upper()} 3D completed successfully")
                else:
                    print(f"⚠️  {method_name.upper()} 3D skipped")
            except Exception as e:
                print(f"❌ {method_name.upper()} 3D failed: {e}")
        
        return self.results_2d, self.results_3d

# =============================================================================
# CLUSTERING ANALYSIS
# =============================================================================

def perform_clustering(embeddings_2d, embeddings_3d, labels, min_cluster_size=3):
    """Perform clustering analysis on both 2D and 3D embeddings"""
    print("🎯 Performing clustering analysis...")
    
    clustering_results_2d = {}
    clustering_results_3d = {}
    
    # 2D clustering
    print("📊 Clustering 2D embeddings...")
    for method_name, embedding in embeddings_2d.items():
        try:
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            cluster_labels = clusterer.fit_predict(embedding)
            
            if len(np.unique(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(embedding, cluster_labels)
            else:
                silhouette_avg = 0.0
            
            clustering_results_2d[method_name] = {
                'labels': cluster_labels,
                'clusterer': clusterer,
                'silhouette_score': float(silhouette_avg),
                'n_clusters': int(len(np.unique(cluster_labels[cluster_labels != -1])))
            }
            
            print(f"✅ {method_name.upper()} 2D: {clustering_results_2d[method_name]['n_clusters']} clusters, "
                  f"silhouette score: {clustering_results_2d[method_name]['silhouette_score']:.3f}")
            
        except Exception as e:
            print(f"❌ 2D Clustering failed for {method_name}: {e}")
    
    # 3D clustering
    print("🎲 Clustering 3D embeddings...")
    for method_name, embedding in embeddings_3d.items():
        try:
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            cluster_labels = clusterer.fit_predict(embedding)
            
            if len(np.unique(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(embedding, cluster_labels)
            else:
                silhouette_avg = 0.0
            
            clustering_results_3d[method_name] = {
                'labels': cluster_labels,
                'clusterer': clusterer,
                'silhouette_score': float(silhouette_avg),
                'n_clusters': int(len(np.unique(cluster_labels[cluster_labels != -1])))
            }
            
            print(f"✅ {method_name.upper()} 3D: {clustering_results_3d[method_name]['n_clusters']} clusters, "
                  f"silhouette score: {clustering_results_3d[method_name]['silhouette_score']:.3f}")
            
        except Exception as e:
            print(f"❌ 3D Clustering failed for {method_name}: {e}")
    
    return clustering_results_2d, clustering_results_3d

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_visualizations(embeddings_2d, embeddings_3d, labels, 
                         clustering_results_2d, clustering_results_3d,
                         background_image_path=None):
    """Create both 2D and 3D visualizations"""
    print("🎨 Creating futuristic visualizations...")
    
    cluster_colors = [
        FUTURISTIC_COLORS['primary'],
        FUTURISTIC_COLORS['secondary'],
        FUTURISTIC_COLORS['accent'],
        FUTURISTIC_COLORS['warning'],
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'
    ]
    
    create_2d_viz(embeddings_2d, labels, clustering_results_2d, cluster_colors, background_image_path)
    create_3d_viz(embeddings_3d, labels, clustering_results_3d, cluster_colors)

def create_2d_viz(embeddings, labels, clustering_results, cluster_colors, background_image_path):
    """Create 2D visualization"""
    print("📊 Creating 2D visualization...")
    
    methods = list(embeddings.keys())
    n_methods = len(methods)
    
    if n_methods == 0:
        print("❌ No 2D methods available")
        return
    
    cols = 2 if n_methods > 1 else 1
    rows = (n_methods + 1) // 2 if n_methods > 1 else 1
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'{method.upper()} 2D' for method in methods],
        specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
    )
    
    for idx, method in enumerate(methods):
        row = idx // cols + 1
        col = idx % cols + 1
        
        embedding = embeddings[method]
        cluster_labels = clustering_results[method]['labels']
        
        hover_text = []
        for i, label in enumerate(labels):
            cluster_id = cluster_labels[i]
            cluster_info = f"Cluster: {cluster_id}" if cluster_id != -1 else "Noise"
            hover_text.append(f"<b>{label}</b><br>{cluster_info}")
        
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_embedding = embedding[mask]
            cluster_hover = [hover_text[i] for i in range(len(hover_text)) if mask[i]]
            
            color = '#666666' if cluster_id == -1 else cluster_colors[cluster_id % len(cluster_colors)]
            name = 'Noise' if cluster_id == -1 else f'Cluster {cluster_id}'
            
            fig.add_trace(
                go.Scatter(
                    x=cluster_embedding[:, 0],
                    y=cluster_embedding[:, 1],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=color,
                        line=dict(width=2, color=FUTURISTIC_COLORS['text']),
                        opacity=0.8
                    ),
                    name=f'{method.upper()} - {name}',
                    text=cluster_hover,
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text=f'{method.upper()} Component 1', row=row, col=col)
        fig.update_yaxes(title_text=f'{method.upper()} Component 2', row=row, col=col)
    
    fig.update_layout(
        title='🚀 ADVANCED DIMENSIONALITY REDUCTION - 2D VISUALIZATION',
        template='plotly_dark',
        plot_bgcolor=FUTURISTIC_COLORS['background'],
        paper_bgcolor=FUTURISTIC_COLORS['background'],
        font=dict(family='Orbitron', color=FUTURISTIC_COLORS['text']),
        height=600 * rows
    )
    
    # Add background image
    if background_image_path and os.path.exists(background_image_path):
        try:
            from PIL import Image
            img = Image.open(background_image_path)
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="paper", yref="paper",
                    x=0, y=1, sizex=1, sizey=1,
                    sizing="stretch", opacity=0.15, layer="below"
                )
            )
        except Exception as e:
            print(f"⚠️  Could not add background image: {e}")
    
    fig.write_html('advanced_viz_2d.html')
    print("💾 2D Visualization saved: advanced_viz_2d.html")

def create_3d_viz(embeddings, labels, clustering_results, cluster_colors):
    """Create 3D visualization"""
    print("🎲 Creating 3D visualization...")
    
    methods = list(embeddings.keys())
    n_methods = len(methods)
    
    if n_methods == 0:
        print("❌ No 3D methods available")
        return
    
    cols = 2 if n_methods > 1 else 1
    rows = (n_methods + 1) // 2 if n_methods > 1 else 1
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'{method.upper()} 3D' for method in methods],
        specs=[[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)]
    )
    
    for idx, method in enumerate(methods):
        row = idx // cols + 1
        col = idx % cols + 1
        
        embedding = embeddings[method]
        cluster_labels = clustering_results[method]['labels']
        
        hover_text = []
        for i, label in enumerate(labels):
            cluster_id = cluster_labels[i]
            cluster_info = f"Cluster: {cluster_id}" if cluster_id != -1 else "Noise"
            hover_text.append(f"<b>{label}</b><br>{cluster_info}")
        
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_embedding = embedding[mask]
            cluster_hover = [hover_text[i] for i in range(len(hover_text)) if mask[i]]
            
            color = '#666666' if cluster_id == -1 else cluster_colors[cluster_id % len(cluster_colors)]
            name = 'Noise' if cluster_id == -1 else f'Cluster {cluster_id}'
            
            fig.add_trace(
                go.Scatter3d(
                    x=cluster_embedding[:, 0],
                    y=cluster_embedding[:, 1],
                    z=cluster_embedding[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color,
                        line=dict(width=2, color=FUTURISTIC_COLORS['text']),
                        opacity=0.8
                    ),
                    name=f'{method.upper()} - {name}',
                    text=cluster_hover,
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title='🎲 ADVANCED DIMENSIONALITY REDUCTION - 3D VISUALIZATION',
        template='plotly_dark',
        paper_bgcolor=FUTURISTIC_COLORS['background'],
        font=dict(family='Orbitron', color=FUTURISTIC_COLORS['text']),
        height=800 * rows
    )
    
    fig.write_html('advanced_viz_3d.html')
    print("💾 3D Visualization saved: advanced_viz_3d.html")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("🚀 ADVANCED DIMENSIONALITY REDUCTION VISUALIZATION SYSTEM")
    print("=" * 60)
    
    start_date = '2015-01-01'
    end_date = '2020-01-01'
    
    try:
        # Extract vectors using autoencoder pipeline
        vector_universe, encoder = extract_autoencoder_vectors_from_pipeline(start_date, end_date)
        
        if not vector_universe:
            print("❌ No vectors extracted. Please check your data.")
            return
        
        # Initialize dimensionality reducer
        reducer = AdvancedDimensionalityReducer()
        
        # Prepare data
        X_scaled, labels = reducer.prepare_data(vector_universe)
        
        # Apply all dimensionality reduction methods (2D and 3D)
        embeddings_2d, embeddings_3d = reducer.apply_all_methods(X_scaled)
        
        if not embeddings_2d and not embeddings_3d:
            print("❌ No methods completed successfully.")
            return
        
        # Perform clustering
        clustering_results_2d, clustering_results_3d = perform_clustering(embeddings_2d, embeddings_3d, labels)
        
        # Create visualizations
        create_visualizations(
            embeddings_2d, embeddings_3d, labels,
            clustering_results_2d, clustering_results_3d,
            background_image_path=BACKGROUND_IMAGE_PATH
        )
        
        # Save results
        results = {
            'vector_universe': convert_numpy_types(vector_universe),
            'embeddings_2d': convert_numpy_types(embeddings_2d),
            'embeddings_3d': convert_numpy_types(embeddings_3d),
            'clustering_results_2d': {
                method: {
                    'labels': convert_numpy_types(data['labels']),
                    'silhouette_score': float(data['silhouette_score']),
                    'n_clusters': int(data['n_clusters'])
                } for method, data in clustering_results_2d.items()
            },
            'clustering_results_3d': {
                method: {
                    'labels': convert_numpy_types(data['labels']),
                    'silhouette_score': float(data['silhouette_score']),
                    'n_clusters': int(data['n_clusters'])
                } for method, data in clustering_results_3d.items()
            }
        }
        
        if not safe_json_dump(results, 'advanced_viz_results.json'):
            print("⚠️  Failed to save JSON results")
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        print("✅ Generated files:")
        print("- advanced_viz_2d.html")
        print("- advanced_viz_3d.html")
        print("- advanced_viz_results.json")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set background image path if available
    background_path = "/home/kisero/Downloads/asdf.jpg"
    if os.path.exists(background_path):
        BACKGROUND_IMAGE_PATH = background_path
        print(f"✅ Background image found: {background_path}")
    
    main() 
