"""
Strategy B Backtest: SMA + RSI + Fear & Greed Index
====================================================
Competition: Trading Bot Competition (Jan 1-14, 2025)
Asset: SOL (Solana)

Building on Strategy A, this adds market sentiment filtering
using the Crypto Fear & Greed Index.

Strategy Rules:
---------------
ENTRY (all conditions must be met):
  1. SMA short > SMA long (golden cross confirmed)
  2. RSI between 30-70 (not overbought/oversold)
  3. Fear & Greed Index < 75 (not extreme greed) [NEW]

EXIT (any condition triggers):
  1. Stop loss: -5%
  2. Take profit: +15%
  3. SMA death cross
  4. RSI > 75 (overbought)
  5. Fear & Greed Index > 80 (extreme greed, take profits) [NEW]

Fear & Greed Index:
-------------------
  0-25:  Extreme Fear (good buying opportunity)
  25-45: Fear
  45-55: Neutral
  55-75: Greed
  75-100: Extreme Greed (consider taking profits)

Data Source: Alternative.me Crypto Fear & Greed Index API
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import os
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION (Optimized from Strategy A)
# =============================================================================

# Strategy parameters (optimized)
SMA_SHORT = 10          # Short-term SMA period
SMA_LONG = 50           # Long-term SMA period (optimized from 30)
RSI_PERIOD = 14         # RSI calculation period
RSI_OVERSOLD = 30       # RSI oversold threshold
RSI_OVERBOUGHT = 70     # RSI overbought threshold

# Fear & Greed thresholds [NEW]
FG_ENTRY_MAX = 75       # Don't enter when F&G above this (extreme greed)
FG_EXIT_THRESHOLD = 80  # Exit when F&G exceeds this

# Risk management (optimized)
STOP_LOSS = 0.05        # Stop loss at -5% (optimized from 7%)
TAKE_PROFIT = 0.15      # Take profit at +15%
TRADE_PCT = 0.10        # Use 10% of capital per trade
COMMISSION = 0.001      # 0.1% commission per trade

# Backtest settings
INITIAL_CAPITAL = 10000  # Starting capital in USD

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_from_csv(filepath):
    """
    Load historical price data from local CSV file.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def fetch_binance_data(symbol='SOLUSDT', interval='1h', days=90):
    """
    Fetch historical kline data from Binance API.
    """
    print(f"Fetching {symbol} {interval} data for last {days} days...")
    
    base_url = "https://api.binance.com/api/v3/klines"
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_klines = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_time,
            'limit': 1000
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            klines = response.json()
            
            if not klines or isinstance(klines, dict):
                break
                
            all_klines.extend(klines)
            current_start = klines[-1][0] + 1
            
        except Exception as e:
            print(f"API Error: {e}")
            break
    
    if not all_klines:
        return None
    
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    print(f"Successfully fetched {len(df)} candles")
    return df


def load_data():
    """
    Load price data from CSV or fetch from API.
    """
    csv_path = 'SOL_1h_data.csv'
    
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        df = load_data_from_csv(csv_path)
        print(f"Loaded {len(df)} candles")
        return df
    else:
        print(f"CSV not found, fetching from Binance...")
        return fetch_binance_data()


# =============================================================================
# FEAR & GREED INDEX [NEW]
# =============================================================================

def fetch_fear_greed_history(limit=0):
    """
    Fetch historical Fear & Greed Index from Alternative.me API.
    
    API returns daily data with:
    - value: 0-100 (0 = Extreme Fear, 100 = Extreme Greed)
    - value_classification: "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    - timestamp: Unix timestamp
    
    Args:
        limit: Number of days to fetch (0 = all available, max ~2000 days)
        
    Returns:
        DataFrame with date index and fear_greed column
    """
    print("Fetching Fear & Greed Index history...")
    
    url = "https://api.alternative.me/fng/"
    params = {'limit': limit, 'format': 'json'}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if 'data' not in data:
            print("Warning: No Fear & Greed data returned")
            return None
        
        records = []
        for item in data['data']:
            records.append({
                'date': datetime.fromtimestamp(int(item['timestamp'])),
                'fear_greed': int(item['value']),
                'classification': item['value_classification']
            })
        
        fg_df = pd.DataFrame(records)
        fg_df.set_index('date', inplace=True)
        fg_df.sort_index(inplace=True)
        
        print(f"Successfully fetched {len(fg_df)} days of Fear & Greed data")
        print(f"Date range: {fg_df.index.min().date()} to {fg_df.index.max().date()}")
        
        return fg_df
        
    except Exception as e:
        print(f"Error fetching Fear & Greed data: {e}")
        return None


def merge_fear_greed(price_df, fg_df):
    """
    Merge Fear & Greed daily data with hourly price data.
    
    Since F&G is daily, we forward-fill to match hourly data.
    Each hour of a day gets the same F&G value as that day.
    
    Args:
        price_df: DataFrame with hourly price data
        fg_df: DataFrame with daily Fear & Greed data
        
    Returns:
        price_df with added fear_greed column
    """
    if fg_df is None:
        print("Warning: No F&G data, using neutral value (50)")
        price_df['fear_greed'] = 50
        return price_df
    
    # Create date column for merging
    price_df = price_df.copy()
    price_df['date'] = price_df.index.date
    
    # Create date index for F&G data
    fg_daily = fg_df[['fear_greed']].copy()
    fg_daily.index = fg_daily.index.date
    
    # Map F&G to each row based on date
    price_df['fear_greed'] = price_df['date'].map(
        lambda d: fg_daily.loc[d, 'fear_greed'] if d in fg_daily.index else np.nan
    )
    
    # Forward fill missing values (weekends, holidays)
    price_df['fear_greed'] = price_df['fear_greed'].ffill()
    
    # Backward fill any remaining NaN at the start
    price_df['fear_greed'] = price_df['fear_greed'].bfill()
    
    # If still NaN (no data at all), use neutral
    price_df['fear_greed'] = price_df['fear_greed'].fillna(50)
    
    # Clean up
    price_df.drop('date', axis=1, inplace=True)
    
    fg_coverage = (price_df['fear_greed'] != 50).sum() / len(price_df) * 100
    print(f"Fear & Greed data coverage: {fg_coverage:.1f}%")
    
    return price_df


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_sma(data, period):
    """Calculate Simple Moving Average."""
    return data.rolling(window=period).mean()


def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index.
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def add_indicators(df):
    """
    Add all technical indicators to the DataFrame.
    """
    df = df.copy()
    
    # Moving averages
    df['sma_short'] = calculate_sma(df['close'], SMA_SHORT)
    df['sma_long'] = calculate_sma(df['close'], SMA_LONG)
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    
    return df


# =============================================================================
# SIGNAL GENERATION [UPDATED with Fear & Greed]
# =============================================================================

def generate_signals(df):
    """
    Generate buy/sell signals based on strategy rules.
    
    Buy signal (1):
        - SMA short crosses above SMA long (golden cross)
        - RSI is between 30-70
        - Fear & Greed < 75 (not extreme greed) [NEW]
        
    Sell signal (-1):
        - SMA short crosses below SMA long (death cross)
        - OR RSI > 75 (overbought)
        - OR Fear & Greed > 80 (extreme greed) [NEW]
    
    Args:
        df: DataFrame with price data, indicators, and fear_greed
        
    Returns:
        DataFrame with signal column
    """
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    
    # SMA crossover detection
    sma_bullish = (df['sma_short'] > df['sma_long']).astype(int)
    sma_cross_up = sma_bullish.diff() == 1    # Golden cross
    sma_cross_down = sma_bullish.diff() == -1  # Death cross
    
    # RSI conditions
    rsi_ok = (df['rsi'] > RSI_OVERSOLD) & (df['rsi'] < RSI_OVERBOUGHT)
    rsi_overbought = df['rsi'] > 75
    
    # Fear & Greed conditions [NEW]
    fg_ok_for_entry = df['fear_greed'] < FG_ENTRY_MAX  # Not extreme greed
    fg_extreme_greed = df['fear_greed'] > FG_EXIT_THRESHOLD  # Time to exit
    
    # Generate signals
    # Buy: golden cross + RSI ok + F&G not extreme greed
    buy_condition = sma_cross_up & rsi_ok & fg_ok_for_entry
    signals.loc[buy_condition, 'signal'] = 1
    
    # Sell: death cross OR RSI overbought OR F&G extreme greed
    sell_condition = sma_cross_down | rsi_overbought | fg_extreme_greed
    signals.loc[sell_condition, 'signal'] = -1
    
    return signals


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class Backtester:
    """
    Backtesting engine with stop loss / take profit support.
    """
    
    def __init__(self, initial_capital=10000, trade_pct=0.1, 
                 stop_loss=0.05, take_profit=0.15, commission=0.001):
        self.initial_capital = initial_capital
        self.trade_pct = trade_pct
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.commission = commission
        
    def run(self, df, signals):
        """Execute backtest simulation."""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        
        trades = []
        equity_curve = []
        
        for i in range(len(df)):
            timestamp = df.index[i]
            price = df['close'].iloc[i]
            signal = signals['signal'].iloc[i]
            fg_value = df['fear_greed'].iloc[i] if 'fear_greed' in df.columns else 50
            
            current_equity = capital + position * price
            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'price': price,
                'position': position,
                'fear_greed': fg_value
            })
            
            # Check stop loss / take profit if holding position
            if position > 0:
                pnl_pct = (price - entry_price) / entry_price
                
                # Stop loss triggered
                if pnl_pct <= -self.stop_loss:
                    sell_value = position * price * (1 - self.commission)
                    pnl = sell_value - (position * entry_price)
                    capital += sell_value
                    
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'STOP_LOSS',
                        'price': price,
                        'quantity': position,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct * 100,
                        'fear_greed': fg_value
                    })
                    position = 0
                    continue
                
                # Take profit triggered
                if pnl_pct >= self.take_profit:
                    sell_value = position * price * (1 - self.commission)
                    pnl = sell_value - (position * entry_price)
                    capital += sell_value
                    
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'TAKE_PROFIT',
                        'price': price,
                        'quantity': position,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct * 100,
                        'fear_greed': fg_value
                    })
                    position = 0
                    continue
            
            # Process signals
            if signal == 1 and position == 0:
                trade_amount = capital * self.trade_pct
                quantity = (trade_amount * (1 - self.commission)) / price
                capital -= trade_amount
                position = quantity
                entry_price = price
                
                trades.append({
                    'timestamp': timestamp,
                    'type': 'BUY',
                    'price': price,
                    'quantity': quantity,
                    'pnl': 0,
                    'pnl_pct': 0,
                    'fear_greed': fg_value
                })
                
            elif signal == -1 and position > 0:
                sell_value = position * price * (1 - self.commission)
                pnl = sell_value - (position * entry_price)
                pnl_pct = (price - entry_price) / entry_price
                capital += sell_value
                
                trades.append({
                    'timestamp': timestamp,
                    'type': 'SELL',
                    'price': price,
                    'quantity': position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100,
                    'fear_greed': fg_value
                })
                position = 0
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        
        final_equity = capital + position * df['close'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        if len(trades_df) > 0:
            closed_trades = trades_df[trades_df['type'] != 'BUY']
            if len(closed_trades) > 0:
                win_trades = closed_trades[closed_trades['pnl'] > 0]
                win_rate = len(win_trades) / len(closed_trades) * 100
                avg_win = win_trades['pnl_pct'].mean() if len(win_trades) > 0 else 0
                lose_trades = closed_trades[closed_trades['pnl'] <= 0]
                avg_loss = lose_trades['pnl_pct'].mean() if len(lose_trades) > 0 else 0
            else:
                win_rate = avg_win = avg_loss = 0
        else:
            win_rate = avg_win = avg_loss = 0
        
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Count F&G triggered exits [NEW]
        fg_exits = 0
        if len(trades_df) > 0 and 'fear_greed' in trades_df.columns:
            # Count sells where F&G was above threshold
            sell_trades = trades_df[trades_df['type'] == 'SELL']
            fg_exits = (sell_trades['fear_greed'] > FG_EXIT_THRESHOLD).sum()
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'fg_triggered_exits': fg_exits,
            'trades': trades_df,
            'equity_curve': equity_df
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_indicators(df, last_n=200):
    """
    Plot price with indicators including Fear & Greed.
    """
    plot_df = df.iloc[-last_n:].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), 
                             gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Price and SMA
    ax1 = axes[0]
    ax1.plot(plot_df.index, plot_df['close'], label='Close', 
             color='#2c3e50', linewidth=1.5)
    ax1.plot(plot_df.index, plot_df['sma_short'], label=f'SMA {SMA_SHORT}', 
             color='#e74c3c', linewidth=1.2)
    ax1.plot(plot_df.index, plot_df['sma_long'], label=f'SMA {SMA_LONG}', 
             color='#3498db', linewidth=1.2)
    
    ax1.fill_between(plot_df.index, plot_df['sma_short'], plot_df['sma_long'],
                     where=plot_df['sma_short'] > plot_df['sma_long'],
                     color='#2ecc71', alpha=0.3, label='Bullish')
    ax1.fill_between(plot_df.index, plot_df['sma_short'], plot_df['sma_long'],
                     where=plot_df['sma_short'] <= plot_df['sma_long'],
                     color='#e74c3c', alpha=0.3, label='Bearish')
    
    ax1.set_title('SOL Price with SMA Crossover', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # RSI
    ax2 = axes[1]
    ax2.plot(plot_df.index, plot_df['rsi'], color='#9b59b6', linewidth=1.5)
    ax2.axhline(y=RSI_OVERBOUGHT, color='#e74c3c', linestyle='--')
    ax2.axhline(y=RSI_OVERSOLD, color='#2ecc71', linestyle='--')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax2.fill_between(plot_df.index, RSI_OVERBOUGHT, 100, color='#e74c3c', alpha=0.2)
    ax2.fill_between(plot_df.index, 0, RSI_OVERSOLD, color='#2ecc71', alpha=0.2)
    ax2.set_title('RSI', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Fear & Greed Index [NEW]
    ax3 = axes[2]
    fg_color = plot_df['fear_greed'].apply(
        lambda x: '#2ecc71' if x < 25 else '#27ae60' if x < 45 else '#f1c40f' if x < 55 
                  else '#e67e22' if x < 75 else '#e74c3c'
    )
    ax3.bar(plot_df.index, plot_df['fear_greed'], color=fg_color, alpha=0.7, width=0.03)
    ax3.axhline(y=FG_ENTRY_MAX, color='#e74c3c', linestyle='--', 
                label=f'Entry Max ({FG_ENTRY_MAX})')
    ax3.axhline(y=FG_EXIT_THRESHOLD, color='#c0392b', linestyle='-', 
                label=f'Exit Threshold ({FG_EXIT_THRESHOLD})')
    ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax3.set_title('Fear & Greed Index', fontsize=14, fontweight='bold')
    ax3.set_ylabel('F&G')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Add zone labels
    ax3.text(plot_df.index[5], 12, 'Extreme Fear', fontsize=9, color='#2ecc71')
    ax3.text(plot_df.index[5], 87, 'Extreme Greed', fontsize=9, color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig('strategy_b_indicators.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved: strategy_b_indicators.png")


def plot_backtest_results(results, df):
    """
    Plot backtest performance with Fear & Greed overlay.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    
    equity_df = results['equity_curve']
    trades_df = results['trades']
    
    # 1. Price with trade markers
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='SOL Price', color='#2c3e50', linewidth=1)
    
    if len(trades_df) > 0:
        buys = trades_df[trades_df['type'] == 'BUY']
        ax1.scatter(buys['timestamp'], buys['price'], marker='^', 
                   color='#2ecc71', s=100, label='Buy', zorder=5)
        
        for exit_type, color in [('SELL', '#3498db'), 
                                  ('STOP_LOSS', '#e74c3c'),
                                  ('TAKE_PROFIT', '#27ae60')]:
            exits = trades_df[trades_df['type'] == exit_type]
            if len(exits) > 0:
                ax1.scatter(exits['timestamp'], exits['price'], marker='v',
                           color=color, s=100, label=exit_type.replace('_', ' ').title(), 
                           zorder=5)
    
    ax1.set_title('Trading Signals (Strategy B: SMA + RSI + Fear & Greed)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Equity curve with F&G background
    ax2 = axes[1]
    ax2.plot(equity_df['timestamp'], equity_df['equity'], 
             color='#3498db', linewidth=1.5, label='Portfolio')
    ax2.axhline(y=results['initial_capital'], color='gray', linestyle='--',
               label=f'Initial: ${results["initial_capital"]:,.0f}')
    ax2.fill_between(equity_df['timestamp'], results['initial_capital'], 
                     equity_df['equity'],
                     where=equity_df['equity'] >= results['initial_capital'],
                     color='#2ecc71', alpha=0.3)
    ax2.fill_between(equity_df['timestamp'], results['initial_capital'], 
                     equity_df['equity'],
                     where=equity_df['equity'] < results['initial_capital'],
                     color='#e74c3c', alpha=0.3)
    ax2.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Portfolio Value (USD)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Fear & Greed over time
    ax3 = axes[2]
    ax3.fill_between(equity_df['timestamp'], 0, equity_df['fear_greed'],
                     color='#f39c12', alpha=0.5)
    ax3.axhline(y=FG_ENTRY_MAX, color='#e74c3c', linestyle='--', label=f'Entry Max ({FG_ENTRY_MAX})')
    ax3.axhline(y=FG_EXIT_THRESHOLD, color='#c0392b', linestyle='-', label=f'Exit ({FG_EXIT_THRESHOLD})')
    ax3.set_title('Fear & Greed Index Over Time', fontsize=14, fontweight='bold')
    ax3.set_ylabel('F&G Index')
    ax3.set_xlabel('Date')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('strategy_b_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved: strategy_b_results.png")


# =============================================================================
# COMPARISON WITH STRATEGY A
# =============================================================================

def run_strategy_a_comparison(df):
    """
    Run Strategy A (without F&G) for comparison.
    """
    # Generate signals without F&G filter
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    
    sma_bullish = (df['sma_short'] > df['sma_long']).astype(int)
    sma_cross_up = sma_bullish.diff() == 1
    sma_cross_down = sma_bullish.diff() == -1
    
    rsi_ok = (df['rsi'] > RSI_OVERSOLD) & (df['rsi'] < RSI_OVERBOUGHT)
    rsi_overbought = df['rsi'] > 75
    
    # No F&G filter for Strategy A
    buy_condition = sma_cross_up & rsi_ok
    signals.loc[buy_condition, 'signal'] = 1
    
    sell_condition = sma_cross_down | rsi_overbought
    signals.loc[sell_condition, 'signal'] = -1
    
    backtester = Backtester(
        initial_capital=INITIAL_CAPITAL,
        trade_pct=TRADE_PCT,
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT,
        commission=COMMISSION
    )
    
    return backtester.run(df, signals)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def print_results(results, strategy_name="Strategy B"):
    """Print formatted backtest results."""
    print("=" * 60)
    print(f"BACKTEST RESULTS: {strategy_name}")
    print("=" * 60)
    print(f"\nCapital:")
    print(f"  Initial:      ${results['initial_capital']:,.2f}")
    print(f"  Final:        ${results['final_equity']:,.2f}")
    print(f"  Total Return: {results['total_return']:+.2f}%")
    
    print(f"\nTrading Stats:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Win Rate:     {results['win_rate']:.1f}%")
    print(f"  Avg Win:      {results['avg_win']:+.2f}%")
    print(f"  Avg Loss:     {results['avg_loss']:.2f}%")
    
    print(f"\nRisk Metrics:")
    print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
    
    if 'fg_triggered_exits' in results:
        print(f"\nFear & Greed Impact:")
        print(f"  F&G Triggered Exits: {results['fg_triggered_exits']}")
    
    print("=" * 60)


def print_comparison(results_a, results_b):
    """Print side-by-side comparison of Strategy A vs B."""
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON: A vs B")
    print("=" * 70)
    print(f"{'Metric':<25} {'Strategy A':>20} {'Strategy B':>20}")
    print("-" * 70)
    print(f"{'Total Return':<25} {results_a['total_return']:>19.2f}% {results_b['total_return']:>19.2f}%")
    print(f"{'Win Rate':<25} {results_a['win_rate']:>19.1f}% {results_b['win_rate']:>19.1f}%")
    print(f"{'Max Drawdown':<25} {results_a['max_drawdown']:>19.2f}% {results_b['max_drawdown']:>19.2f}%")
    print(f"{'Total Trades':<25} {results_a['total_trades']:>20} {results_b['total_trades']:>20}")
    print(f"{'Avg Win':<25} {results_a['avg_win']:>19.2f}% {results_b['avg_win']:>19.2f}%")
    print(f"{'Avg Loss':<25} {results_a['avg_loss']:>19.2f}% {results_b['avg_loss']:>19.2f}%")
    print("=" * 70)
    
    # Determine winner
    if results_b['total_return'] > results_a['total_return']:
        print("✅ Strategy B outperforms Strategy A!")
    else:
        print("⚠️ Strategy A still better - may need to tune F&G thresholds")


def main():
    """Main execution function."""
    print("=" * 60)
    print("STRATEGY B BACKTEST")
    print("SMA + RSI + Fear & Greed Index")
    print("=" * 60)
    
    # Step 1: Load price data
    print("\n[1/6] Loading price data...")
    df = load_data()
    if df is None:
        print("ERROR: Could not load data")
        return
    print(f"  Data range: {df.index.min()} to {df.index.max()}")
    
    # Step 2: Fetch Fear & Greed data
    print("\n[2/6] Fetching Fear & Greed Index...")
    fg_df = fetch_fear_greed_history(limit=0)  # Get all available data
    
    # Step 3: Calculate indicators and merge F&G
    print("\n[3/6] Calculating indicators...")
    df = add_indicators(df)
    df = merge_fear_greed(df, fg_df)
    
    print(f"  SMA Short ({SMA_SHORT}): {df['sma_short'].iloc[-1]:.2f}")
    print(f"  SMA Long ({SMA_LONG}): {df['sma_long'].iloc[-1]:.2f}")
    print(f"  RSI: {df['rsi'].iloc[-1]:.2f}")
    print(f"  Fear & Greed: {df['fear_greed'].iloc[-1]:.0f}")
    
    # Step 4: Generate signals
    print("\n[4/6] Generating signals...")
    signals = generate_signals(df)
    buy_count = (signals['signal'] == 1).sum()
    sell_count = (signals['signal'] == -1).sum()
    print(f"  Buy signals: {buy_count}")
    print(f"  Sell signals: {sell_count}")
    
    # Step 5: Run backtest
    print("\n[5/6] Running backtest...")
    backtester = Backtester(
        initial_capital=INITIAL_CAPITAL,
        trade_pct=TRADE_PCT,
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT,
        commission=COMMISSION
    )
    results_b = backtester.run(df, signals)
    print_results(results_b, "Strategy B (with Fear & Greed)")
    
    # Run Strategy A for comparison
    print("\nRunning Strategy A for comparison...")
    results_a = run_strategy_a_comparison(df)
    print_results(results_a, "Strategy A (without Fear & Greed)")
    
    # Print comparison
    print_comparison(results_a, results_b)
    
    # Step 6: Visualize
    print("\n[6/6] Generating charts...")
    plot_indicators(df)
    plot_backtest_results(results_b, df)
    
    print("\n" + "=" * 60)
    print("STRATEGY B PARAMETERS (for bot.js)")
    print("=" * 60)
    print(f"""
// Strategy B: SMA + RSI + Fear & Greed
const SMA_SHORT_PERIOD = {SMA_SHORT};
const SMA_LONG_PERIOD = {SMA_LONG};

const RSI_PERIOD = {RSI_PERIOD};
const RSI_OVERSOLD = {RSI_OVERSOLD};
const RSI_OVERBOUGHT = {RSI_OVERBOUGHT};

// Fear & Greed thresholds
const FG_ENTRY_MAX = {FG_ENTRY_MAX};      // Don't buy above this
const FG_EXIT_THRESHOLD = {FG_EXIT_THRESHOLD};  // Sell when above this

// Risk management
const STOP_LOSS = {STOP_LOSS};
const TAKE_PROFIT = {TAKE_PROFIT};
const TRADE_PERCENTAGE = {TRADE_PCT};
""")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
