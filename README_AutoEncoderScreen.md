# 🚀 AutoEncoder Pairs Screener Dashboard

## Overview
The AutoEncoder Pairs Screener is a sophisticated application that uses a trained autoencoder model to identify cointegrated stock pairs and visualize their spreads and ratios through an interactive futuristic dashboard.

## Features

### 🧠 Core Functionality
- **Trained Encoder Loading**: Loads the pre-trained autoencoder from `encoder.pkl`
- **Same Pipeline Processing**: Uses identical preprocessing pipeline as the training model
- **Cointegration Testing**: Applies statistical tests to find the most cointegrated pairs
- **Top 10 Pairs**: Automatically identifies and displays the top 10 most cointegrated stock pairs

### 📊 Spread & Ratio Calculations
- **Linear Regression Beta**: Calculates beta coefficient using linear regression
- **Spread Calculation**: `stock1 - beta * stock2`
- **Ratio Calculation**: `stock1 / stock2`
- **Statistical Metrics**: Provides mean, std, and other statistics

### 🎮 Interactive Dashboard Features
- **Zoomable Charts**: Full zoom, pan, and selection capabilities
- **Multiple Display Options**:
  - Individual stock prices
  - Spread visualization with zero reference line
  - Ratio visualization with mean reference line
- **Real-time Updates**: Dynamic chart updates based on user selections

### 🎨 Futuristic Design
- **Custom Backgrounds**: 4 different futuristic themes
  - Dark Matrix: Deep black with cyan accents
  - Neon Grid: Dark blue with neon highlights
  - Space Blue: Space-themed blue gradient
  - Cyber Purple: Cyberpunk purple theme
- **Glowing Effects**: Text shadows and neon borders
- **Modern UI**: Glass morphism effects and gradient backgrounds

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Update Directory Path
Edit the `DIRECTORY` variable in `AutoEncoderScreen.py` to match your stock data location:
```python
DIRECTORY = '/path/to/your/stock_data/'  # Update this path
```

### 3. Ensure Encoder File Exists
Make sure `encoder.pkl` exists in the same directory (created by running the training script first).

## Usage

### Basic Execution
```bash
python AutoEncoderScreen.py
```

### What Happens When You Run It:

1. **Encoder Loading**: The application loads the trained encoder from `encoder.pkl`
2. **Stock Processing**: Processes all stock files through the same preprocessing pipeline:
   - Rolling z-score normalization with LOOKBACK window
   - Stride windowing for sequence creation
   - Autoencoder encoding to latent vectors
3. **Clustering**: Uses HDBSCAN to cluster similar stocks
4. **Cointegration Testing**: Tests all clustered pairs for cointegration
5. **Dashboard Launch**: Opens interactive dashboard at `http://127.0.0.1:8050`

### Dashboard Controls

#### 🎛️ Control Panel
- **Pair Selector**: Choose from top 10 cointegrated pairs (sorted by p-value)
- **Display Options**:
  - ☐ Show Spread
  - ☐ Show Ratio  
  - ☐ Show Individual Stocks
- **Background Selector**: Choose from 4 futuristic themes

#### 📈 Chart Features
- **Interactive Zoom**: Mouse wheel or selection tools
- **Hover Information**: Detailed data points on hover
- **Legend Toggle**: Click legend items to show/hide traces
- **Multiple Subplots**: Automatically arranged based on display options

#### 📊 Statistics Panel
Shows real-time statistics for selected pair:
- Stock names and beta coefficient
- P-value and cointegration strength
- Spread statistics (mean, standard deviation)
- Ratio statistics and data point count

## Configuration Options

### Time Period (in `main()` function)
```python
start_date = '2020-01-01'  # Screening start date
end_date = '2024-01-01'    # Screening end date
alpha = 0.05               # Cointegration significance level
```

### Preprocessing Parameters
```python
LOOKBACK = 256   # Must match training lookback
stride = 64      # Window stride for processing
```

## Mathematical Details

### Spread Calculation
The spread is calculated using linear regression to find the optimal hedge ratio:

1. **Beta Calculation**: `β = (X'X)⁻¹X'y` where X = stock2, y = stock1
2. **Spread Formula**: `Spread = Stock1 - β × Stock2`

### Cointegration Testing
- **Pairwise**: Uses Phillips-Ouliaris test for 2 stocks
- **Multivariate**: Uses Johansen test for 3+ stocks
- **Alpha Threshold**: Filters pairs by statistical significance

### Clustering Process
1. **Vector Generation**: Encodes stock sequences to latent vectors
2. **HDBSCAN Clustering**: Groups similar behavioral patterns
3. **Pair Generation**: Creates all combinations within clusters
4. **Statistical Filtering**: Tests and ranks by cointegration strength

## Troubleshooting

### Common Issues

#### "encoder.pkl not found"
- **Solution**: Run the training script (`AutoEncoderPairScreenerV2.py`) first

#### "No valid stock data found"
- **Solution**: Check `DIRECTORY` path and ensure CSV files exist
- **Format**: Ensure CSV files have 'Close' column and date index

#### "No cointegrated pairs found"
- **Solution**: Increase `alpha` threshold (try 0.1 or 0.2)
- **Alternative**: Check data quality and time period coverage

#### Dashboard not loading
- **Solution**: Ensure all dependencies are installed
- **Port Issue**: Try changing port in `app.run_server(port=8051)`

### Performance Tips
- **Large Datasets**: Reduce time period or increase stride for faster processing
- **Memory Usage**: Monitor memory usage with many stocks
- **Processing Speed**: Use SSD storage for faster file I/O

## Output Interpretation

### Spread Analysis
- **Zero Line**: Ideal convergence point for mean reversion
- **Volatility**: Higher spread volatility indicates stronger mean reversion opportunities
- **Trending**: Persistent trends may indicate structural changes

### Ratio Analysis
- **Mean Line**: Historical average ratio between stocks
- **Extremes**: Ratios far from mean may indicate trading opportunities
- **Stability**: Stable ratios suggest strong cointegration relationship

### P-Values
- **< 0.01**: Very strong cointegration
- **< 0.05**: Strong cointegration (default threshold)
- **< 0.10**: Moderate cointegration
- **> 0.10**: Weak or no cointegration

## Advanced Features

### Custom Background Implementation
The dashboard supports custom backgrounds through CSS injection:
```css
body {
    background: your-custom-gradient;
}
```

### Extensibility
The modular design allows easy extension:
- Add new cointegration tests
- Implement additional statistical measures
- Create custom visualization components

## Support
For issues or feature requests, check:
1. Data file formats and paths
2. Dependencies installation
3. Training script completion
4. Console output for detailed error messages 