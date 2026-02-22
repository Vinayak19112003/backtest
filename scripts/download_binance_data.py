import os
import requests
import zipfile
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import io

def download_data():
    symbol = "BTCUSDT"
    interval = "15m"
    base_url = "https://data.binance.vision/data/spot/monthly/klines"

    end_date = datetime.now()
    start_date = end_date - relativedelta(years=3)

    # Set to first day of the month for start_date
    current_date = datetime(year=start_date.year, month=start_date.month, day=1)

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    all_dfs = []

    print(f"Downloading 15m data for {symbol} from {current_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")

    while current_date <= end_date:
        month_str = current_date.strftime("%Y-%m")
        url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{month_str}.zip"
        print(f"Downloading {url}...")
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as f:
                        cols = [
                            "open_time", "open", "high", "low", "close", "volume",
                            "close_time", "quote_volume", "count",
                            "taker_buy_volume", "taker_buy_quote_volume", "ignore"
                        ]
                        df = pd.read_csv(f, names=cols)
                        all_dfs.append(df)
            else:
                print(f"No zip found for {month_str} (status code: {response.status_code}). This is expected for the current/incomplete month.")
        except Exception as e:
            print(f"Error downloading {month_str}: {e}")
        
        current_date += relativedelta(months=1)

    if all_dfs:
        print("Concatenating data...")
        final_df = pd.concat(all_dfs, ignore_index=True)
        print("Converting timestamps...")
        # Force numeric, dropping any headers or malformed rows that might be in the CSV
        final_df['open_time'] = pd.to_numeric(final_df['open_time'], errors='coerce')
        final_df['close_time'] = pd.to_numeric(final_df['close_time'], errors='coerce')
        
        # Filter out clear anomalies (e.g. open_time > year 2030 in ms)
        # Year 2030 in ms is ~1.9e12
        final_df = final_df[final_df['open_time'] < 2000000000000]
        final_df = final_df.dropna(subset=['open_time', 'close_time'])
        
        final_df['open_time'] = pd.to_datetime(final_df['open_time'], unit='ms', errors='coerce')
        final_df['close_time'] = pd.to_datetime(final_df['close_time'], unit='ms', errors='coerce')
        final_df = final_df.dropna(subset=['open_time', 'close_time'])
        
        output_path = os.path.join(data_dir, f"{symbol}_{interval}_3_years.csv")
        print("Saving to CSV...")
        final_df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(final_df)} rows to {output_path}")
    else:
        print("No data downloaded!")

if __name__ == '__main__':
    download_data()
