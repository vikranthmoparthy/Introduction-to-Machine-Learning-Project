"""
In this file, we normalize all the data according to the z-distribution and define the target variable (1 or 0). 
To implement normalization easily, we use sklearn's StandardScaler.
The source documentation for the scaler: 
    sklearn's StandardScaler: https://shorturl.at/BJyWe
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    input_filename = "daily_energy_data_clean.csv" 
    output_filename = "processed_training_data.csv"
    
    df = pd.read_csv(input_filename) #We load the daily data
    
    #Definition of target variable: 1 if price went up that day, 0 if it went down (or stayed the same).
    df['target'] = (df['close_price'] > df['open_price']).astype(int) 
    
    #Defining features to scale, which includes everything except "date" and "target".
    features_to_scale = [
        'avg_load', 
        'max_solar', 
        'avg_wind_onshore', 
        'avg_temp', 
        'gas_price', 
        'price_yesterday'
    ]
    
    scaler = StandardScaler()
    
    #This function does two things: calculates the mean and std for each feature AND also applies the z-transform to each value
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    df.to_csv(output_filename, index=False) #Save

if __name__ == "__main__":
    main()