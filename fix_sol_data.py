"""
Fix SOL Data
============
Fixes the column name mismatch between original and new data.

Original data: timestamp, Open, High, Low, Close
New data: timestamp, open, high, low, close, volume

This script merges them correctly.
"""

import pandas as pd

def fix_sol_data():
    print("=" * 60)
    print("FIX SOL DATA")
    print("=" * 60)
    
    # Load the messy file
    print("\n[1] Loading current SOL_1h_data.csv...")
    df = pd.read_csv('SOL_1h_data.csv')
    print(f"    Columns: {df.columns.tolist()}")
    print(f"    Rows: {len(df)}")
    
    # Check which columns have data
    print("\n[2] Analyzing columns...")
    
    # For each row, use lowercase columns if available, else uppercase
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create clean columns (prefer lowercase/new data, fallback to uppercase/old data)
    print("\n[3] Merging columns...")
    
    # open
    if 'open' in df.columns and 'Open' in df.columns:
        df['open_clean'] = df['open'].fillna(df['Open'])
    elif 'open' in df.columns:
        df['open_clean'] = df['open']
    elif 'Open' in df.columns:
        df['open_clean'] = df['Open']
    
    # high
    if 'high' in df.columns and 'High' in df.columns:
        df['high_clean'] = df['high'].fillna(df['High'])
    elif 'high' in df.columns:
        df['high_clean'] = df['high']
    elif 'High' in df.columns:
        df['high_clean'] = df['High']
    
    # low
    if 'low' in df.columns and 'Low' in df.columns:
        df['low_clean'] = df['low'].fillna(df['Low'])
    elif 'low' in df.columns:
        df['low_clean'] = df['low']
    elif 'Low' in df.columns:
        df['low_clean'] = df['Low']
    
    # close
    if 'close' in df.columns and 'Close' in df.columns:
        df['close_clean'] = df['close'].fillna(df['Close'])
    elif 'close' in df.columns:
        df['close_clean'] = df['close']
    elif 'Close' in df.columns:
        df['close_clean'] = df['Close']
    
    # volume (might only exist in new data)
    if 'volume' in df.columns:
        df['volume_clean'] = df['volume'].fillna(0)
    else:
        df['volume_clean'] = 0
    
    # Create final clean DataFrame
    df_clean = pd.DataFrame({
        'timestamp': df['timestamp'],
        'open': df['open_clean'],
        'high': df['high_clean'],
        'low': df['low_clean'],
        'close': df['close_clean'],
        'volume': df['volume_clean']
    })
    
    # Remove duplicates (keep latest)
    df_clean = df_clean.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Sort by timestamp
    df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
    
    # Check for NaN
    nan_count = df_clean['close'].isna().sum()
    print(f"    NaN values in close: {nan_count}")
    
    if nan_count > 0:
        print("    Removing rows with NaN...")
        df_clean = df_clean.dropna(subset=['close'])
    
    print(f"\n[4] Final data:")
    print(f"    Rows: {len(df_clean)}")
    print(f"    Date range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
    print(f"    Start price: ${df_clean['close'].iloc[0]:.2f}")
    print(f"    End price: ${df_clean['close'].iloc[-1]:.2f}")
    
    # Save
    output_file = 'SOL_1h_data.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"\n✅ Saved to {output_file}")
    
    # Show sample
    print("\n[5] Sample data:")
    print("\nFirst 5 rows:")
    print(df_clean.head().to_string(index=False))
    print("\nLast 5 rows:")
    print(df_clean.tail().to_string(index=False))
    
    return df_clean


if __name__ == "__main__":
    df = fix_sol_data()