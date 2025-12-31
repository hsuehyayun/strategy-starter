import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import sys


def _interval_to_resolution_minutes(interval: str) -> int:
    mapping = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}. Supported: {', '.join(mapping.keys())}")
    return mapping[interval]


def fetch_pyth_history(symbol: str, resolution_minutes: int, from_timestamp_s: int, to_timestamp_s: int):
    """Fetch OHLC candlestick data from Pyth. Returns DataFrame or None."""
    url = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
    params = {
        "symbol": symbol,
        "resolution": str(resolution_minutes),
        "from": int(from_timestamp_s),
        "to": int(to_timestamp_s),
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("s") != "ok" or not data.get("t"):
            return None

        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data["t"], unit='s'),
            'Open': pd.to_numeric(data["o"], errors='coerce'),
            'High': pd.to_numeric(data["h"], errors='coerce'),
            'Low': pd.to_numeric(data["l"], errors='coerce'),
            'Close': pd.to_numeric(data["c"], errors='coerce'),
        })
        return df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Pyth: {e}")
        return None


def fetch_all_historical_data(symbol, interval, days_back):
    """Fetch all historical OHLC data from Pyth over the desired range."""
    end_time_dt = datetime.now()
    start_time_dt = end_time_dt - timedelta(days=days_back)
    start_s = int(start_time_dt.timestamp())
    end_s = int(end_time_dt.timestamp())
    resolution_min = _interval_to_resolution_minutes(interval)
    candle_interval_s = resolution_min * 60

    print(f"Fetching {symbol} data from {start_time_dt.strftime('%Y-%m-%d')} to {end_time_dt.strftime('%Y-%m-%d')}")
    print(f"Interval: {interval} ({resolution_min}m)")
    print(f"Expected candles: ~{(end_s - start_s) // candle_interval_s}")
    print("-" * 60)

    all_data = []
    max_bars_per_request = 5000
    window_s = max_bars_per_request * candle_interval_s
    current_start_s = start_s

    while current_start_s < end_s:
        df_batch = fetch_pyth_history(
            symbol=symbol,
            resolution_minutes=resolution_min,
            from_timestamp_s=current_start_s,
            to_timestamp_s=min(end_s, current_start_s + window_s),
        )

        if df_batch is None or len(df_batch) == 0:
            current_start_s += window_s + candle_interval_s
            continue

        all_data.append(df_batch)
        last_s = int(df_batch['timestamp'].iloc[-1].timestamp())
        print(f"Request {len(all_data)}: Fetched {len(df_batch)} candles (up to {df_batch['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')})")
        current_start_s = last_s + candle_interval_s
        time.sleep(0.25)

    if not all_data:
        print("No data fetched")
        return None

    final_df = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    print("-" * 60)
    print(f"✓ Total candles fetched: {len(final_df)}")
    print(f"✓ Date range: {final_df['timestamp'].iloc[0]} to {final_df['timestamp'].iloc[-1]}")
    return final_df


def save_to_csv(df, filename):
    """Save DataFrame to CSV with OHLC only"""
    df_to_save = df.copy().set_index('timestamp')
    df_to_save.to_csv(filename)
    print(f"\n✓ Data saved to: {filename}")
    print(f"  Format: timestamp (index), Open, High, Low, Close")
    print(f"\nFirst 3 rows:")
    print(df_to_save.head(3))
    return filename


if __name__ == "__main__":
    SYMBOL = "Crypto.SOL/USD"
    INTERVAL = '1h'
    DAYS_BACK = 365 * 5
    OUTPUT_FILE = f'SOL_{INTERVAL}_data.csv'
    
    print("=" * 60)
    print("Pyth Historical Data Fetcher (OHLC only)")
    print("=" * 60)
    
    df = fetch_all_historical_data(
        symbol=SYMBOL,
        interval=INTERVAL,
        days_back=DAYS_BACK
    )
    
    if df is not None:
        save_to_csv(df, OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("Data Summary:")
        print("=" * 60)
        print(df.describe())
        
        print("\n✓ Success! Your data is ready for backtesting.")
    else:
        print("\n✗ Failed to fetch data.")
        sys.exit(1)


