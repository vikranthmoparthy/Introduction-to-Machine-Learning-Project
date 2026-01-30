"""
In this script, we perform data aggregation by taking our hourly data and squashing it down into a single daily summary row (1 row per day).
To write this script, we had to explore additional Pandas functionality including groupby and label-based indexing.
Sources:
    Pandas groupby: https://shorturl.at/Xnuc5
    Pandas indexing: https://shorturl.at/UpZOe
"""

import pandas as pd

def main():
    filename = "nl_energy_data_2025.csv"
    output_filename = "daily_energy_data_clean.csv"
        
    df = pd.read_csv(filename, index_col=0, parse_dates=True) #Reading the hourly data

    daily_rows = [] #Intializing list to hold summarized daily records

    #We split the DataFrame into chunks, where each chunk is one specific day. Then, we iterate through every day (group) one by one.
    for date, group in df.groupby(df.index.date): 

        #Before summarizing, we must verify this specific day has a start (00:00) and an end (23:00)
        #Otherwise, we will get skewed data if a day is incomplete
        hours = group.index.hour
        if 0 in hours and 23 in hours:
            
            #Extract open and close prices
            p_open = group.loc[group.index.hour == 0, 'price_eur'].values[0]
            p_close = group.loc[group.index.hour == 23, 'price_eur'].values[0]
            
            #Compress 24 rows into 1 dict
            row = {
                'date': date,
                'open_price': p_open, #Price at start of day
                'close_price': p_close, #Price at end of day
                'avg_load': group['Forecasted Load'].mean(), #Average demand
                'max_solar': group['Solar'].max(), #Peak solar production
                'avg_wind_onshore': group['Wind Onshore'].mean(),
                'avg_temp': group['temp_c'].mean(),

                #Since gas prices were already the same for every hour of the day, we just use the first one
                'gas_price': group['gas_price'].iloc[0]
            }
            daily_rows.append(row)
            
    df_clean = pd.DataFrame(daily_rows) #Convert lists of dicts to a proper dataframe

    #Creating the lag feature: We use the previous day's closing price as a predictor for today.
    df_clean['price_yesterday'] = df_clean['close_price'].shift(1)

    df_clean.dropna(inplace=True) #Drop the first row (which has NaN for price-yesterday)

    df_clean.to_csv(output_filename, index=False) #Save

if __name__ == "__main__":
    main()