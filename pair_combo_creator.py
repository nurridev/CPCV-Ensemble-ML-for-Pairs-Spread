import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
OUTPUT_FILE = 'combined_pairs.csv'
# Writes to stock_data.csv based n amount of pairs: Creates N column with price adjusted price series
df_stock_data = pd.read_csv('stock_data.csv')
df_stock_data.set_index(df_stock_data.columns[0])

df_pair_data  =  pd.read_csv('pair_database.csv')
df_pair_data.sort_values('strength',ascending=True,inplace=True)
output_df = pd.DataFrame()
# DataFrame File, Num of pairs, Lookback of Hedge Ratio
def pair_combo_gen(df_pair_data, df_stock_data, n):
    #Accepts csv file with close data (any length) creates synthetic assets based on dynamic hedge ratio
    counter= 0 
    for row_idx, row_data in df_pair_data.iterrows():
        pair1 = row_data['Pair1']
        pair2 = row_data['Pair2']
        pair_name = pair1 + ' ' + pair2  
        print(pair1 + ' ' + pair2)
        Lprice1 = df_stock_data[pair1]
        Lprice2 = df_stock_data[pair2]
        # used for statiistics
        hedge_ratios = []
    
        combined_price_lst    = []
        # Lookback for dynamic hedge rati
        window = 200
        for i in range(window,len(Lprice1)):
            price_win1 = np.log(Lprice1[i-window:i])
            price_win2 = np.log(Lprice2[i-window:i]) 
            # Add constants to price_win because it wont work without it like tf            
            price_win1 = sm.add_constant(price_win1)
            model = sm.OLS(price_win2, price_win1).fit()
            alpha, beta = model.params
            hedge_ratios.append(beta)  # coefficient of X (slope)
             
            spread = Lprice2[i] - (beta * Lprice1[i])

            combined_price_lst.append(spread)
        # Combine df
        output_df[pair_name] = combined_price_lst
        counter += 1 
        if counter == n:
            break
    #Output final df
    output_df.to_csv(OUTPUT_FILE)

pair_combo_gen(df_pair_data,df_stock_data,10)
