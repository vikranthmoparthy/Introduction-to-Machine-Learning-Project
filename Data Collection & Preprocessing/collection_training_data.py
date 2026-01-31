"""
This file implements the data pipeline, which involves extracting, transforming and loading required data. 

To be able to implement the pipeline, we had to learn parts of the documentation for the ENTSO-E and openmeteo API clients.
The sources where we learnt the functionality are: 
    Open-Meteo Client: https://shorturl.at/vfYVq
    ENTSOE Client (third-party wrapper): https://shorturl.at/13ZAx

Additionally, we had to learn some Pandas and requests functionality, such as time-series, requests cache and retry-requests
The sources are:
    Pandas Time series: https://shorturl.at/EC7ER
    Pandas join, concatenating and merging functionality: https://tinyurl.com/38dkpwrw
    retry-requests: https://shorturl.at/F4Kgn
    requests cache: https://shorturl.at/lttlj

Additionally, the source for the gas prices:
    Gas Prices: https://tinyurl.com/yc2vzb9v
"""

import pandas as pd
import openmeteo_requests #Python client for openmeteo-API (weather data)
import requests_cache
from retry_requests import retry
from entsoe import EntsoePandasClient #For extracting ENTSOE data (transmission / electricity)
                                        #We used a third-party wrapper rather than the official client for ease of functionality; it allows data to be
                                        #requested directly as Pandas data frames

API_KEY = "8af5be55-2363-4b7e-9dc4-a7a59b850367"
COUNTRY = "NL"
YEAR = 2025
LAT, LON = 52.0907, 5.1214 #Cooordinates for Utrecht
GAS_FILE = "gas_prices.csv"

current_date = pd.Timestamp.now(tz="Europe/Amsterdam") #Define the time range relative to now
START = pd.Timestamp(f"{YEAR}-01-01", tz="Europe/Amsterdam") #Setting start data 
END = current_date - pd.Timedelta(days=1) #Setting end-date as the day before the data was extracted (also ensure data is for full day)

def main():
    #Fetch the raw data from the three sources (one of them is local) 
    df_energy = get_entsoe_data()
    df_weather = get_weather_data()
    s_gas = get_gas_data()

    #Validation is necessary, since there could be many possible reasons for errors at runtime 
    if df_energy is None or df_weather is None or s_gas is None:
        print("Data generation FAILURE")
    else:
        try:
            #Merging hourly data between energy and weather
            df_final = df_energy.join(df_weather, how='inner') 

            #Creating the lag feature (explained in report)
            df_final['price_yesterday'] = df_final['price_eur'].shift(24)

            #Merging Daily Gas Data onto hourly energy data
            df_final['date_str'] = df_final.index.strftime('%Y-%m-%d')
            s_gas.index = s_gas.index.strftime('%Y-%m-%d')
            
            #Map the daily gas price to every hour of the respective day
            df_final = df_final.merge(s_gas, left_on='date_str', right_index=True, how='left')

            #Handle missing gas data
            df_final['gas_price'] = df_final['gas_price'].ffill()

            #Cleanup by removing empty rows and temporary string columns
            df_final.drop(columns=['date_str'], inplace=True)
            df_final.dropna(inplace=True)

            if not df_final.empty:
                filename = f"nl_energy_data_{YEAR}.csv"
                df_final.to_csv(filename)
                print(f"CSV Generated") #Save the file and output success
            else:
                print("Merged dataset is empty.")
        except Exception:
            print(f"Merge Error")

def get_entsoe_data(): #For fetching electricity data from the ENTSO-E platform.
    try: #We catch specific errors, so we can pinpoint the reason behind the error at runtime
        client = EntsoePandasClient(api_key=API_KEY)
        
        #Query three different datasets for the same defined timeframe
        prices = client.query_day_ahead_prices(COUNTRY, start=START, end=END)
        load = client.query_load_forecast(COUNTRY, start=START, end=END)
        gen = client.query_wind_and_solar_forecast(COUNTRY, start=START, end=END)

        df = pd.DataFrame(prices, columns=['price_eur']) #convert series to a dataframe

        #Joining Load and Generation data.
        df = df.join(load, rsuffix='_load')
        df = df.join(gen, rsuffix='_gen')
        
        df.index = df.index.tz_convert('UTC')#Important: we standardized timezone to UTC to match the Weather API
        return df
    except Exception as e:
        print(f"ENTSO-E Error")
        return None

def get_weather_data(): #For fetching historical weather from Open-Meteo. Here we use caching to avoid getting banned for too many requests (as happened with YFinance).
    try:
        #We setup local cache so we don't re-download data if we run the script twice
        cache = requests_cache.CachedSession('.cache', expire_after=3600) 
        #We wrap cache with up to 5 retries and exponential backoff time between retries; 0.2 was chosen, as per convention
        retry_client = retry(cache, retries=5, backoff_factor=0.2) 
        openmeteo = openmeteo_requests.Client(session=retry_client)

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": START.strftime("%Y-%m-%d"),
            "end_date": END.strftime("%Y-%m-%d"),
            "hourly": ["temperature_2m", "wind_speed_100m", "direct_normal_irradiance"]
        }
        #Fetch data
        res = openmeteo.weather_api(url, params=params)[0]
        hourly = res.Hourly()

        #We need to manually reconstruct the timestamps since openmeteo returns a start time and an interval, not a list of dates.
        dates = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
        #Create dataframe from numpy arrays
        df = pd.DataFrame({
            'temp_c': hourly.Variables(0).ValuesAsNumpy(),
            'wind_speed_100m': hourly.Variables(1).ValuesAsNumpy(),
            'solar_dni': hourly.Variables(2).ValuesAsNumpy()
        }, index=dates)
        
        return df
    except Exception as e:
        print(f"Weather Error")
        return None

def get_gas_data(): #Reads a local CSV for containing gas prices
    try:
        df = pd.read_csv(GAS_FILE, usecols=['Date', 'Price'], quotechar='"') #Reading the CSV
        
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y') #Parsing the US format dates
        df.set_index('Date', inplace=True)
        
        series = df['Price']
        series.name = 'gas_price'

        #We remove the timezone info to allow comparison with the start/end variables
        series.index = series.index.tz_localize(None)

        #Filter for only the dates within our defined global start/end range
        mask = (series.index >= START.tz_localize(None)) & (series.index <= END.tz_localize(None))
        return series.loc[mask]
        
    except Exception as e:
        print(f"CSV Parse Error")
        return None

if __name__ == "__main__":
    main()