import pandas as pd

def main():
    filename = "nl_energy_data_2025.csv"
    output_filename = "daily_energy_data.csv"
        
    df = pd.read_csv(filename, index_col=0, parse_dates=True)

    daily_rows = []

    for date, group in df.groupby(df.index.date):
        
        hours = group.index.hour
        if 0 in hours and 23 in hours:
            
            p_open = group.loc[group.index.hour == 0, 'price_eur'].values[0]
            p_close = group.loc[group.index.hour == 23, 'price_eur'].values[0]
            
            row = {
                'date': date,
                'open_price': p_open,
                'close_price': p_close,
                'avg_load': group['Forecasted Load'].mean(),
                'max_solar': group['Solar'].max(),
                'avg_wind_onshore': group['Wind Onshore'].mean(),
                'avg_temp': group['temp_c'].mean(),
                'gas_price': group['gas_price'].iloc[0] 
            }
            daily_rows.append(row)
            
    df_clean = pd.DataFrame(daily_rows)
    df_clean.to_csv(output_filename, index=False)

if __name__ == "__main__":
    main()