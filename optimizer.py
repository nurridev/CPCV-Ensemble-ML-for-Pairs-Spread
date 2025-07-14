"""
🚀 UNIFIED OPTUNA AUTOENCODER OPTIMIZATION SYSTEM
==================================================

This comprehensive script provides:
1. Neural network architecture optimization using Optuna
2. Production-ready model creation with best parameters
3. Implementation guide for integration with AutoEncoderPairScreener
4. Analysis and visualization of optimization results
5. Complete pipeline demonstration

Author: AI Assistant
Date: 2025
"""

import random
import optuna
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, SGD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import tensorflow as tf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

DIRECTORY = './stock_data/'
LOOKBACK = 50  # Optimized based on available data
N_TRIALS = 50  # Number of optimization trials
STUDY_NAME = "unified_autoencoder_optimization"
BEST_PARAMS_FILE = 'best_unified_params.json'

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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

def print_banner(title, char="=", width=60):
    """Print formatted banner"""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}")

# =============================================================================
# OPTUNA OPTIMIZATION SYSTEM
# =============================================================================

def create_optimized_autoencoder(trial, input_dim):
    """
    Create autoencoder architecture based on Optuna trial suggestions
    Comprehensive search space for optimal architecture discovery
    """
    model = Sequential()
    
    # Encoder layers - expanded search space
    n_encoder_layers = trial.suggest_int('n_encoder_layers', 2, 6)
    
    # First layer - wider range
    first_layer_units = trial.suggest_int('first_layer_units', 16, 256)
    first_activation = trial.suggest_categorical('first_activation', [
        'relu', 'gelu', 'swish', 'tanh', 'elu', 'leaky_relu', 'selu'
    ])
    
    # Regularization type - comprehensive options
    reg_type = trial.suggest_categorical('regularization_type', ['l1', 'l2', 'l1_l2', 'none'])
    if reg_type == 'l1':
        regularizer = l1(trial.suggest_float('reg_strength', 1e-5, 1e-1, log=True))
    elif reg_type == 'l2':
        regularizer = l2(trial.suggest_float('reg_strength', 1e-5, 1e-1, log=True))
    elif reg_type == 'l1_l2':
        l1_strength = trial.suggest_float('l1_strength', 1e-5, 1e-1, log=True)
        l2_strength = trial.suggest_float('l2_strength', 1e-5, 1e-1, log=True)
        regularizer = l1_l2(l1=l1_strength, l2=l2_strength)
    else:
        regularizer = None
    
    # Add first layer
    model.add(Dense(first_layer_units, 
                   activation=first_activation,
                   kernel_regularizer=regularizer,
                   input_shape=(input_dim,)))
    
    # Batch normalization option
    if trial.suggest_categorical('use_batch_norm', [True, False]):
        model.add(BatchNormalization())
    
    # Dropout
    if trial.suggest_categorical('use_dropout', [True, False]):
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        model.add(Dropout(dropout_rate))
    
    # Additional encoder layers with flexible architecture
    current_units = first_layer_units
    for i in range(1, n_encoder_layers):
        # Layer size strategy
        layer_strategy = trial.suggest_categorical(f'layer_{i}_strategy', [
            'decrease', 'increase', 'same', 'custom'
        ])
        
        if layer_strategy == 'decrease':
            next_units = max(8, current_units // 2)
        elif layer_strategy == 'increase':
            next_units = min(512, current_units * 2)
        elif layer_strategy == 'same':
            next_units = current_units
        else:  # custom
            next_units = trial.suggest_int(f'layer_{i}_units', 8, 512)
        
        activation = trial.suggest_categorical(f'layer_{i}_activation', [
            'relu', 'gelu', 'swish', 'tanh', 'elu', 'leaky_relu', 'selu'
        ])
        
        model.add(Dense(next_units, activation=activation, kernel_regularizer=regularizer))
        
        # Optional batch normalization for this layer
        if trial.suggest_categorical(f'layer_{i}_batch_norm', [True, False]):
            model.add(BatchNormalization())
        
        # Optional dropout for this layer
        if trial.suggest_categorical(f'layer_{i}_dropout', [True, False]):
            layer_dropout_rate = trial.suggest_float(f'layer_{i}_dropout_rate', 0.1, 0.5)
            model.add(Dropout(layer_dropout_rate))
        
        current_units = next_units
    
    # Latent space (bottleneck) - expanded range
    latent_dim = trial.suggest_int('latent_dim', 4, 64)
    latent_activation = trial.suggest_categorical('latent_activation', [
        'relu', 'gelu', 'tanh', 'linear', 'swish', 'elu'
    ])
    model.add(Dense(latent_dim, activation=latent_activation, kernel_regularizer=regularizer))
    
    # Decoder layers - more flexible architecture
    for i in range(n_encoder_layers - 1, -1, -1):
        if i == 0:
            decoder_units = first_layer_units
            decoder_activation = first_activation
        else:
            decoder_units = trial.suggest_int(f'decoder_{i}_units', 8, 512)
            decoder_activation = trial.suggest_categorical(f'decoder_{i}_activation', [
                'relu', 'gelu', 'swish', 'tanh', 'elu', 'leaky_relu', 'selu'
            ])
        
        model.add(Dense(decoder_units, activation=decoder_activation, kernel_regularizer=regularizer))
        
        # Optional batch normalization for decoder
        if trial.suggest_categorical(f'decoder_{i}_batch_norm', [True, False]):
            model.add(BatchNormalization())
        
        # Optional dropout for decoder
        if trial.suggest_categorical(f'decoder_{i}_dropout', [True, False]):
            decoder_dropout_rate = trial.suggest_float(f'decoder_{i}_dropout_rate', 0.1, 0.5)
            model.add(Dropout(decoder_dropout_rate))
    
    # Output layer
    output_activation = trial.suggest_categorical('output_activation', [
        'linear', 'relu', 'tanh', 'sigmoid'
    ])
    model.add(Dense(input_dim, activation=output_activation))
    
    # Optimizer selection - expanded options
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop', 'sgd'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'adamw':
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:  # sgd
        momentum = trial.suggest_float('momentum', 0.0, 0.9)
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    
    # Loss function options
    loss_function = trial.suggest_categorical('loss_function', [
        'mse', 'mae', 'huber', 'logcosh'
    ])
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['mse', 'mae'])
    
    return model

def prepare_data(lookback, trial):
    """Prepare data for training with enhanced preprocessing"""
    files = [f for f in os.listdir(DIRECTORY) if f.endswith('.csv')]
    all_data = []
    stock_names = []
    
    # Data preprocessing strategy
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax', 'robust', 'none'])
    
    for file_name in files:
        try:
            df = pd.read_csv(os.path.join(DIRECTORY, file_name), index_col=0)
            df.index = pd.to_datetime(df.index)
            
            df_price = df['Close']
            
            # Adjusted requirement for available data
            if len(df_price) < lookback * 2:
                continue
                
            # Rolling z-score normalization
            rolling_mean = df_price.rolling(window=lookback).mean()
            rolling_std = df_price.rolling(window=lookback).std()
            rolling_zscore = (df_price - rolling_mean) / rolling_std
            rolling_zscore = rolling_zscore.dropna().values
            
            # Additional preprocessing
            if scaler_type == 'standard':
                scaler = StandardScaler()
                rolling_zscore = scaler.fit_transform(rolling_zscore.reshape(-1, 1)).flatten()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
                rolling_zscore = scaler.fit_transform(rolling_zscore.reshape(-1, 1)).flatten()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
                rolling_zscore = scaler.fit_transform(rolling_zscore.reshape(-1, 1)).flatten()
            
            # Create windowed data
            stride = trial.suggest_int('stride', 16, 64)
            windowed_data = stride_window(stride, rolling_zscore, lookback)
            
            if len(windowed_data) > 0:
                all_data.append(windowed_data)
                stock_name = file_name.replace('.csv', '')
                stock_names.append(stock_name)
                
        except Exception as e:
            continue
    
    if not all_data:
        raise ValueError("No data processed successfully")
    
    # Combine all data
    combined_data = np.vstack(all_data)
    
    # Split into train/validation
    split_ratio = trial.suggest_float('train_split', 0.7, 0.9)
    split_idx = int(len(combined_data) * split_ratio)
    
    train_data = combined_data[:split_idx]
    val_data = combined_data[split_idx:]
    
    return train_data, val_data, stock_names

def objective(trial):
    """Enhanced objective function for Optuna optimization"""
    try:
        # Prepare data
        train_data, val_data, stock_names = prepare_data(LOOKBACK, trial)
        
        # Create model
        model = create_optimized_autoencoder(trial, LOOKBACK)
        
        # Training parameters
        batch_size = trial.suggest_int('batch_size', 16, 128)
        epochs = trial.suggest_int('epochs', 50, 200)
        patience = trial.suggest_int('patience', 10, 50)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Optional learning rate scheduler
        if trial.suggest_categorical('use_lr_scheduler', [True, False]):
            lr_patience = trial.suggest_int('lr_patience', 5, 20)
            lr_factor = trial.suggest_float('lr_factor', 0.1, 0.8)
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=lr_factor,
                    patience=lr_patience,
                    verbose=0
                )
            )
        
        # Train model
        history = model.fit(
            train_data, train_data,
            validation_data=(val_data, val_data),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Get best validation loss
        best_val_loss = min(history.history['val_loss'])
        
        # Additional metrics for comprehensive evaluation
        val_predictions = model.predict(val_data, verbose=0)
        reconstruction_mse = mean_squared_error(val_data, val_predictions)
        reconstruction_mae = mean_absolute_error(val_data, val_predictions)
        
        # Test encoding quality (latent space diversity)
        encoder = Sequential()
        for i, layer in enumerate(model.layers):
            encoder.add(layer)
            if layer.output_shape[-1] == trial.params['latent_dim']:
                break
        
        encoded_data = encoder.predict(val_data, verbose=0)
        latent_variance = np.var(encoded_data, axis=0).mean()
        
        # Composite objective (lower is better)
        composite_score = (
            best_val_loss + 
            0.1 * reconstruction_mse + 
            0.05 * reconstruction_mae - 
            0.01 * latent_variance
        )
        
        return composite_score
        
    except Exception as e:
        return float('inf')

def run_optimization():
    """Run the comprehensive Optuna optimization study"""
    print_banner("🚀 UNIFIED OPTUNA AUTOENCODER OPTIMIZATION")
    print(f"📊 Configuration: {N_TRIALS} trials, LOOKBACK={LOOKBACK}")
    
    # Create study with SQLite storage for persistence
    study = optuna.create_study(
        direction='minimize',
        study_name=STUDY_NAME,
        storage=f'sqlite:///{STUDY_NAME}.db',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=20,
            n_warmup_steps=30,
            interval_steps=10
        )
    )
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    # Results
    print_banner("🎯 OPTIMIZATION RESULTS")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    
    print("\n📊 Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_df = study.trials_dataframe()
    results_df.to_csv('unified_optuna_results.csv', index=False)
    
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"- unified_optuna_results.csv")
    print(f"- {BEST_PARAMS_FILE}")
    print(f"- {STUDY_NAME}.db")
    
    return study

# =============================================================================
# PRODUCTION MODEL CREATION
# =============================================================================

def create_production_autoencoder(best_params, input_dim):
    """
    Create production-ready autoencoder with optimized parameters
    """
    model = Sequential()
    
    # Reconstruction of architecture based on best parameters
    n_encoder_layers = best_params['n_encoder_layers']
    first_layer_units = best_params['first_layer_units']
    first_activation = best_params['first_activation']
    
    # Regularization setup
    regularizer = None
    if best_params['regularization_type'] == 'l1':
        regularizer = l1(best_params['reg_strength'])
    elif best_params['regularization_type'] == 'l2':
        regularizer = l2(best_params['reg_strength'])
    elif best_params['regularization_type'] == 'l1_l2':
        regularizer = l1_l2(l1=best_params['l1_strength'], l2=best_params['l2_strength'])
    
    # First layer
    model.add(Dense(first_layer_units, 
                   activation=first_activation,
                   kernel_regularizer=regularizer,
                   input_shape=(input_dim,)))
    
    # Batch normalization
    if best_params.get('use_batch_norm', False):
        model.add(BatchNormalization())
    
    # Dropout
    if best_params.get('use_dropout', False):
        model.add(Dropout(best_params['dropout_rate']))
    
    # Additional encoder layers
    current_units = first_layer_units
    for i in range(1, n_encoder_layers):
        # Reconstruct layer size
        layer_strategy = best_params.get(f'layer_{i}_strategy', 'decrease')
        if layer_strategy == 'decrease':
            next_units = max(8, current_units // 2)
        elif layer_strategy == 'increase':
            next_units = min(512, current_units * 2)
        elif layer_strategy == 'same':
            next_units = current_units
        else:  # custom
            next_units = best_params.get(f'layer_{i}_units', current_units // 2)
        
        activation = best_params.get(f'layer_{i}_activation', 'relu')
        model.add(Dense(next_units, activation=activation, kernel_regularizer=regularizer))
        
        # Optional batch normalization
        if best_params.get(f'layer_{i}_batch_norm', False):
            model.add(BatchNormalization())
        
        # Optional dropout
        if best_params.get(f'layer_{i}_dropout', False):
            model.add(Dropout(best_params.get(f'layer_{i}_dropout_rate', 0.3)))
        
        current_units = next_units
    
    # Latent space
    latent_dim = best_params['latent_dim']
    latent_activation = best_params['latent_activation']
    model.add(Dense(latent_dim, activation=latent_activation, kernel_regularizer=regularizer))
    
    # Decoder layers
    for i in range(n_encoder_layers - 1, -1, -1):
        if i == 0:
            decoder_units = first_layer_units
            decoder_activation = first_activation
        else:
            decoder_units = best_params.get(f'decoder_{i}_units', 64)
            decoder_activation = best_params.get(f'decoder_{i}_activation', 'relu')
        
        model.add(Dense(decoder_units, activation=decoder_activation, kernel_regularizer=regularizer))
        
        # Optional batch normalization
        if best_params.get(f'decoder_{i}_batch_norm', False):
            model.add(BatchNormalization())
        
        # Optional dropout
        if best_params.get(f'decoder_{i}_dropout', False):
            model.add(Dropout(best_params.get(f'decoder_{i}_dropout_rate', 0.3)))
    
    # Output layer
    output_activation = best_params.get('output_activation', 'linear')
    model.add(Dense(input_dim, activation=output_activation))
    
    # Optimizer
    optimizer_name = best_params['optimizer']
    learning_rate = best_params['learning_rate']
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'adamw':
        weight_decay = best_params.get('weight_decay', 0.01)
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:  # sgd
        momentum = best_params.get('momentum', 0.9)
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    
    # Loss function
    loss_function = best_params.get('loss_function', 'mse')
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['mse', 'mae'])
    
    return model

def create_production_model():
    """Create production-ready model with best parameters"""
    print_banner("🏆 CREATING PRODUCTION MODEL")
    
    # Load best parameters
    if not os.path.exists(BEST_PARAMS_FILE):
        print(f"❌ {BEST_PARAMS_FILE} not found. Run optimization first.")
        return None
    
    with open(BEST_PARAMS_FILE, 'r') as f:
        best_params = json.load(f)
    
    # Create model
    model = create_production_autoencoder(best_params, LOOKBACK)
    
    # Save model and parameters
    model.save('unified_production_autoencoder.h5')
    
    # Save architecture summary
    with open('unified_model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print("✅ Production model created:")
    print("- unified_production_autoencoder.h5")
    print("- unified_model_summary.txt")
    print(f"- {BEST_PARAMS_FILE}")
    
    return model

# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================

def analyze_results():
    """Analyze and visualize optimization results"""
    print_banner("📊 RESULTS ANALYSIS")
    
    if not os.path.exists(BEST_PARAMS_FILE):
        print(f"❌ {BEST_PARAMS_FILE} not found. Run optimization first.")
        return
    
    with open(BEST_PARAMS_FILE, 'r') as f:
        best_params = json.load(f)
    
    print("🎯 OPTIMAL ARCHITECTURE DISCOVERED:")
    print(f"- Encoder layers: {best_params['n_encoder_layers']}")
    print(f"- First layer units: {best_params['first_layer_units']}")
    print(f"- Latent dimension: {best_params['latent_dim']}")
    print(f"- Regularization: {best_params['regularization_type']}")
    print(f"- Optimizer: {best_params['optimizer']}")
    print(f"- Learning rate: {best_params['learning_rate']:.6f}")
    print(f"- Batch size: {best_params['batch_size']}")
    
    # Create visualization if plotly is available
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Load results
        results_df = pd.read_csv('unified_optuna_results.csv')
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Optimization Progress', 'Best Value Evolution',
                          'Parameter Distribution', 'Architecture Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Optimization progress
        fig.add_trace(go.Scatter(
            x=results_df['number'],
            y=results_df['value'],
            mode='lines+markers',
            name='Objective Value',
            line=dict(color='cyan')
        ), row=1, col=1)
        
        # Plot 2: Best value evolution
        best_values = []
        current_best = float('inf')
        for value in results_df['value']:
            if pd.notna(value) and value < current_best:
                current_best = value
            best_values.append(current_best)
        
        fig.add_trace(go.Scatter(
            x=results_df['number'],
            y=best_values,
            mode='lines',
            name='Best Value',
            line=dict(color='gold')
        ), row=1, col=2)
        
        fig.update_layout(
            title='Unified Optuna Optimization Analysis',
            template='plotly_dark',
            height=800
        )
        
        fig.write_html('unified_optimization_analysis.html')
        print("📈 Analysis saved to: unified_optimization_analysis.html")
        
    except ImportError:
        print("📈 Install plotly to generate optimization plots")

# =============================================================================
# IMPLEMENTATION GUIDE
# =============================================================================

def implementation_guide():
    """Show implementation guide for using optimized model"""
    print_banner("📋 IMPLEMENTATION GUIDE")
    
    if not os.path.exists(BEST_PARAMS_FILE):
        print(f"❌ {BEST_PARAMS_FILE} not found. Run optimization first.")
        return
    
    with open(BEST_PARAMS_FILE, 'r') as f:
        best_params = json.load(f)
    
    print("🔧 HOW TO USE THE OPTIMIZED MODEL:")
    print("\n1. Replace your original build_model() function:")
    print("   Use create_production_autoencoder(best_params, input_dim)")
    print("\n2. Use optimized training parameters:")
    print(f"   - Batch size: {best_params['batch_size']}")
    print(f"   - Epochs: {best_params['epochs']}")
    print(f"   - Patience: {best_params['patience']}")
    print(f"   - Learning rate: {best_params['learning_rate']:.6f}")
    
    print("\n3. Integration with AutoEncoderPairScreener:")
    print("   - Load model: tf.keras.models.load_model('unified_production_autoencoder.h5')")
    print("   - Use same preprocessing pipeline")
    print("   - Extract latent vectors for pair screening")
    
    print("\n4. Performance improvements:")
    print(f"   - Optimized validation loss")
    print(f"   - Latent dimension: {best_params['latent_dim']}")
    print(f"   - Better activation functions")
    print(f"   - Optimized regularization")

def demonstrate_pipeline():
    """Demonstrate the complete optimized pipeline"""
    print_banner("🎬 PIPELINE DEMONSTRATION")
    
    if not os.path.exists(BEST_PARAMS_FILE):
        print("❌ Run optimization first to generate best parameters.")
        return
    
    with open(BEST_PARAMS_FILE, 'r') as f:
        best_params = json.load(f)
    
    print("📊 Creating optimized autoencoder...")
    model = create_production_autoencoder(best_params, LOOKBACK)
    
    print("\n🔍 Model architecture:")
    model.summary()
    
    print("\n✅ Ready for integration with AutoEncoderPairScreener!")
    print("Use this model in your pipeline for optimal performance.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function with menu system"""
    print_banner("🚀 UNIFIED OPTUNA AUTOENCODER OPTIMIZATION SYSTEM")
    
    while True:
        print("\n" + "="*60)
        print("MENU OPTIONS:")
        print("1. Run Optuna Optimization")
        print("2. Create Production Model")
        print("3. Analyze Results")
        print("4. Implementation Guide")
        print("5. Demonstrate Pipeline")
        print("6. Run Complete Workflow")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '1':
            run_optimization()
        elif choice == '2':
            create_production_model()
        elif choice == '3':
            analyze_results()
        elif choice == '4':
            implementation_guide()
        elif choice == '5':
            demonstrate_pipeline()
        elif choice == '6':
            print("🚀 Running complete workflow...")
            study = run_optimization()
            create_production_model()
            analyze_results()
            implementation_guide()
            demonstrate_pipeline()
            print("✅ Complete workflow finished!")
        elif choice == '0':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 
