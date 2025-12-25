import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from entsoe import EntsoePandasClient

API_KEY = "8af5be55-2363-4b7e-9dc4-a7a59b850367"
COUNTRY = "NL"
YEAR = 2025
LAT, LON = 52.0907, 5.1214
GAS_FILE = "gas_prices.csv"

current_date = pd.Timestamp.now(tz="Europe/Amsterdam")
START = pd.Timestamp(f"{YEAR}-01-01", tz="Europe/Amsterdam")
END = current_date - pd.Timedelta(days=1)

def main():
    df_energy = get_entsoe_data()
    df_weather = get_weather_data()
    s_gas = get_gas_data()

    if df_energy is None or df_weather is None or s_gas is None:
        print("Data generation FAILURE")
    else:
        try:
            df_final = df_energy.join(df_weather, how='inner')

            df_final['date_str'] = df_final.index.strftime('%Y-%m-%d')
            s_gas.index = s_gas.index.strftime('%Y-%m-%d')
            
            df_final = df_final.merge(s_gas, left_on='date_str', right_index=True, how='left')
            df_final['gas_price'] = df_final['gas_price'].ffill()

            df_final.drop(columns=['date_str'], inplace=True)
            df_final.dropna(inplace=True)

            if not df_final.empty:
                filename = f"nl_energy_data_{YEAR}.csv"
                df_final.to_csv(filename)
                print("CSV Generated")
            else:
                print("Merged dataset is empty.")
        except Exception as e:
            print(f"Merge Error: {e}")

def get_entsoe_data():
    try:
        client = EntsoePandasClient(api_key=API_KEY)
        
        prices = client.query_day_ahead_prices(COUNTRY, start=START, end=END)
        load = client.query_load_forecast(COUNTRY, start=START, end=END)
        gen = client.query_wind_and_solar_forecast(COUNTRY, start=START, end=END)

        df = pd.DataFrame(prices, columns=['price_eur'])
        df = df.join(load, rsuffix='_load')
        df = df.join(gen, rsuffix='_gen')
        
        df.index = df.index.tz_convert('UTC')
        return df
    except Exception as e:
        print(f"ENTSO-E Error: {e}")
        return None

def get_weather_data():
    try:
        cache = requests_cache.CachedSession('.cache', expire_after=3600)
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

        res = openmeteo.weather_api(url, params=params)[0]
        hourly = res.Hourly()
        
        dates = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
        
        df = pd.DataFrame({
            'temp_c': hourly.Variables(0).ValuesAsNumpy(),
            'wind_speed_100m': hourly.Variables(1).ValuesAsNumpy(),
            'solar_dni': hourly.Variables(2).ValuesAsNumpy()
        }, index=dates)
        
        return df
    except Exception as e:
        print(f"Weather Error: {e}")
        return None

def get_gas_data():
    try:
        df = pd.read_csv(GAS_FILE, usecols=['Date', 'Price'], quotechar='"')
        
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df.set_index('Date', inplace=True)
        
        series = df['Price']
        series.name = 'gas_price'
        series.index = series.index.tz_localize(None)
        
        mask = (series.index >= START.tz_localize(None)) & (series.index <= END.tz_localize(None))
        return series.loc[mask]
        
    except Exception as e:
        print(f"CSV Parse Error: {e}")
        return None

if __name__ == "__main__":
    main()