import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    input_filename = "daily_energy_data_clean.csv" 
    output_filename = "processed_training_data.csv"
    
    df = pd.read_csv(input_filename)
    
    df['target'] = (df['close_price'] > df['open_price']).astype(int)
    
    features_to_scale = [
        'avg_load', 
        'max_solar', 
        'avg_wind_onshore', 
        'avg_temp', 
        'gas_price', 
        'price_yesterday'
    ]
    
    scaler = StandardScaler()
    
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    df.to_csv(output_filename, index=False)

if __name__ == "__main__":
    main()