"""
SOL Data Downloader
===================
Downloads hourly SOL/USDT data from Binance API.

Binance API is FREE and doesn't require authentication for market data.

Usage:
    python3 download_sol_data.py

Output:
    SOL_1h_data_updated.csv
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Trading pair
SYMBOL = "SOLUSDT"

# Interval (1h = 1 hour)
INTERVAL = "1h"

# How far back to fetch (in days)
DAYS_BACK = 1200  # ~3.3 years

# Output filename
OUTPUT_FILE = "SOL_1h_data_updated.csv"

# Binance API endpoint
BASE_URL = "https://api.binance.com/api/v3/klines"

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests

# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def fetch_klines(symbol, interval, start_time, end_time=None, limit=1000):
    """
    Fetch candlestick data from Binance.
    
    Args:
        symbol: Trading pair (e.g., "SOLUSDT")
        interval: Candle interval (e.g., "1h", "1d")
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds (optional)
        limit: Max candles per request (max 1000)
        
    Returns:
        List of candle data
    """
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'limit': limit
    }
    
    if end_time:
        params['endTime'] = end_time
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  API Error: {response.status_code}")
            print(f"  Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"  Request error: {e}")
        return []


def download_all_data(symbol, interval, days_back):
    """
    Download all historical data by paginating through API.
    
    Args:
        symbol: Trading pair
        interval: Candle interval
        days_back: How many days of history to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {symbol} {interval} data...")
    print(f"  Fetching last {days_back} days...")
    
    all_data = []
    
    # Calculate time range
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    
    current_start = start_time
    
    while current_start < end_time:
        data = fetch_klines(symbol, interval, current_start, end_time)
        
        if not data:
            print("  No more data or error occurred")
            break
        
        all_data.extend(data)
        
        # Update progress
        current_date = datetime.fromtimestamp(data[-1][0] / 1000)
        print(f"  Fetched up to {current_date.strftime('%Y-%m-%d %H:%M')} ({len(all_data)} candles)")
        
        # Move start time forward
        current_start = data[-1][0] + 1
        
        # Rate limiting
        time.sleep(REQUEST_DELAY)
    
    if not all_data:
        print("  No data fetched!")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Process columns
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Keep only essential columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n  Total candles: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def merge_with_existing(new_df, existing_file):
    """
    Merge new data with existing CSV file.
    
    Args:
        new_df: New DataFrame
        existing_file: Path to existing CSV
        
    Returns:
        Merged DataFrame
    """
    if os.path.exists(existing_file):
        print(f"\nMerging with existing file: {existing_file}")
        
        existing_df = pd.read_csv(existing_file)
        
        if 'timestamp' in existing_df.columns:
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
        
        # Combine
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates (keep latest)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Sort
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        print(f"  Existing: {len(existing_df)} rows")
        print(f"  New: {len(new_df)} rows")
        print(f"  Combined: {len(combined)} rows")
        
        return combined
    
    return new_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main download function."""
    print("=" * 60)
    print("SOL DATA DOWNLOADER")
    print("=" * 60)
    print(f"\nSymbol: {SYMBOL}")
    print(f"Interval: {INTERVAL}")
    print(f"Days back: {DAYS_BACK}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Download data
    df = download_all_data(SYMBOL, INTERVAL, DAYS_BACK)
    
    if len(df) == 0:
        print("\nFailed to download data!")
        return
    
    # Try to merge with existing data
    if os.path.exists('SOL_1h_data.csv'):
        df = merge_with_existing(df, 'SOL_1h_data.csv')
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved to {OUTPUT_FILE}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total candles: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Start price: ${df['close'].iloc[0]:.2f}")
    print(f"End price: ${df['close'].iloc[-1]:.2f}")
    print(f"Min price: ${df['close'].min():.2f}")
    print(f"Max price: ${df['close'].max():.2f}")
    
    # Show last few rows
    print("\nLatest data:")
    print(df.tail(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"1. Rename {OUTPUT_FILE} to SOL_1h_data.csv (or update your scripts)")
    print("2. Re-run EDA: python3 eda_analysis.py")
    print("3. Re-run optimization: python3 optimize_simplified.py")
    
    return df


if __name__ == "__main__":
    df = main()
