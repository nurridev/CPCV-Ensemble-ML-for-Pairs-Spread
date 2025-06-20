import numpy as np
import pandas as pd
import yfinance as yf
import os
df_final = []
OUTPUT_DIR = 'labeled_data'
data_dir = 'combined_pairs.csv'
df = pd.read_csv(data_dir)
df.reset_index(inplace=True)
print(df.head(5))

# Delete this if fixed
df.set_index(df.columns[0], inplace=True)

# Accepts series only with any index and price series as the first column
def triple_bar(ticker, df, look_forward_barrier, std_mult, std_lookback):
    # Convert series to dataframe 
    df = pd.DataFrame(df)
    # Barrier calculations merge with data
    rolling_std = df.rolling(window=std_lookback).std()
    df['Standard_dev'] = rolling_std 
    df = df.fillna(method='bfill') # backfill STDs
    counter = 0
    df['label'] = np.nan
    # Horizonta barrier calculations
    for index, row in df.iterrows():
        current_price = row[ticker]
        upper_barrier = current_price + std_mult * row['Standard_dev']
        lower_barrier = current_price - std_mult * row['Standard_dev']
        # iterate to all rows after if the index + look_forward_barrier exists
        if counter < len(df) - look_forward_barrier:
            for i in range(1,look_forward_barrier + 1):
                check_price = df.iloc[counter + i][ticker]
                # Break upper barrier?
                if upper_barrier <= check_price:
                    df.iloc[counter, 2] = 1
                    break
                # Break lower barrier?
                if lower_barrier >= check_price:
                    df.iloc[counter, 2] = -1
                    break
            if np.isnan(df.iloc[counter]['label']):
                df.iloc[counter, 2] = 0
            counter +=1
    return df

for col_name, col_data in df.items():
    # look forward, std_mult, std_lookback 
    df = triple_bar(col_name, col_data, 20, 1.5, 20)
    filepath = os.path.join(OUTPUT_DIR, f'{col_name}.csv')
    print(df['label'].value_counts())
    print(col_name)
    df.to_csv(filepath)









