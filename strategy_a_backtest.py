"""
Strategy A Backtest: SMA + RSI + Stop Loss / Take Profit
=========================================================
Competition: Trading Bot Competition (Jan 1-14, 2025)
Asset: SOL (Solana)

Strategy Rules:
---------------
ENTRY (all conditions must be met):
  1. SMA short > SMA long (golden cross confirmed)
  2. RSI between 30-70 (not overbought/oversold)

EXIT (any condition triggers):
  1. Stop loss: -7%
  2. Take profit: +15%
  3. SMA death cross
  4. RSI > 75 (overbought)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Strategy parameters
SMA_SHORT = 10          # Short-term SMA period
SMA_LONG = 30           # Long-term SMA period
RSI_PERIOD = 14         # RSI calculation period
RSI_OVERSOLD = 30       # RSI oversold threshold
RSI_OVERBOUGHT = 70     # RSI overbought threshold

# Risk management
STOP_LOSS = 0.07        # Stop loss at -7%
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
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with OHLCV data indexed by timestamp
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    
    # Handle different possible column names
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    # Ensure numeric types
    for col in ['open', 'high', 'low', 'lose', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def fetch_binance_data(symbol='SOLUSDT', interval='1h', days=90):
    """
    Fetch historical kline data from Binance API.
    
    Args:
        symbol: Trading pair symbol
        interval: Candlestick interval (1m, 5m, 1h, 1d, etc.)
        days: Number of days of historical data
        
    Returns:
        DataFrame with OHLCV data
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
    
    # Convert to DataFrame
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
    Load data from CSV or fetch from API.
    
    Returns:
        DataFrame with OHLCV data
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
# TECHNICAL INDICATORS
# =============================================================================

def calculate_sma(data, period):
    """
    Calculate Simple Moving Average.
    
    Args:
        data: Price series
        period: Number of periods for averaging
        
    Returns:
        Series of SMA values
    """
    return data.rolling(window=period).mean()


def calculate_ema(data, period):
    """
    Calculate Exponential Moving Average.
    
    Args:
        data: Price series
        period: Number of periods
        
    Returns:
        Series of EMA values
    """
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index.
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    
    Args:
        data: Price series
        period: RSI period (default 14)
        
    Returns:
        Series of RSI values (0-100)
    """
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    # Calculate average gain/loss using EMA
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def add_indicators(df):
    """
    Add all technical indicators to the DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    
    # Moving averages
    df['sma_short'] = calculate_sma(df['close'], SMA_SHORT)
    df['sma_long'] = calculate_sma(df['close'], SMA_LONG)
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    
    return df


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(df):
    """
    Generate buy/sell signals based on strategy rules.
    
    Buy signal (1):
        - SMA short crosses above SMA long (golden cross)
        - RSI is between oversold and overbought levels
        
    Sell signal (-1):
        - SMA short crosses below SMA long (death cross)
        - OR RSI exceeds 75 (overbought)
    
    Args:
        df: DataFrame with price data and indicators
        
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
    rsi_overbought = df['rsi'] > 75  # Using 75 instead of 70 for buffer
    
    # Generate signals
    # Buy: golden cross + RSI in normal range
    buy_condition = sma_cross_up & rsi_ok
    signals.loc[buy_condition, 'signal'] = 1
    
    # Sell: death cross OR RSI overbought
    sell_condition = sma_cross_down | rsi_overbought
    signals.loc[sell_condition, 'signal'] = -1
    
    return signals


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class Backtester:
    """
    Backtesting engine with support for:
    - Stop loss / take profit
    - Position sizing
    - Commission fees
    - Performance metrics
    """
    
    def __init__(self, initial_capital=10000, trade_pct=0.1, 
                 stop_loss=0.07, take_profit=0.15, commission=0.001):
        """
        Initialize backtester with parameters.
        
        Args:
            initial_capital: Starting capital in USD
            trade_pct: Percentage of capital to use per trade
            stop_loss: Stop loss percentage (e.g., 0.07 = 7%)
            take_profit: Take profit percentage (e.g., 0.15 = 15%)
            commission: Commission per trade (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.trade_pct = trade_pct
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.commission = commission
        
    def run(self, df, signals):
        """
        Execute backtest simulation.
        
        Args:
            df: DataFrame with price data
            signals: DataFrame with trading signals
            
        Returns:
            Dictionary containing backtest results
        """
        capital = self.initial_capital
        position = 0        # Current position quantity
        entry_price = 0     # Entry price for current position
        
        trades = []         # Trade history
        equity_curve = []   # Portfolio value over time
        
        for i in range(len(df)):
            timestamp = df.index[i]
            price = df['close'].iloc[i]
            signal = signals['signal'].iloc[i]
            
            # Calculate current equity
            current_equity = capital + position * price
            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'price': price,
                'position': position
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
                        'pnl_pct': pnl_pct * 100
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
                        'pnl_pct': pnl_pct * 100
                    })
                    position = 0
                    continue
            
            # Process signals
            if signal == 1 and position == 0:  # Buy signal, no position
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
                    'pnl_pct': 0
                })
                
            elif signal == -1 and position > 0:  # Sell signal, has position
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
                    'pnl_pct': pnl_pct * 100
                })
                position = 0
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        
        final_equity = capital + position * df['close'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        # Win rate and average P&L
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
        
        # Maximum drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'trades': trades_df,
            'equity_curve': equity_df
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_indicators(df, last_n=200):
    """
    Plot price with indicators.
    
    Args:
        df: DataFrame with price and indicators
        last_n: Number of recent candles to display
    """
    plot_df = df.iloc[-last_n:].copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # Price and SMA
    ax1 = axes[0]
    ax1.plot(plot_df.index, plot_df['close'], label='Close', 
             color='#2c3e50', linewidth=1.5)
    ax1.plot(plot_df.index, plot_df['sma_short'], label=f'SMA {SMA_SHORT}', 
             color='#e74c3c', linewidth=1.2)
    ax1.plot(plot_df.index, plot_df['sma_long'], label=f'SMA {SMA_LONG}', 
             color='#3498db', linewidth=1.2)
    
    # Fill between SMAs
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
    ax2.axhline(y=RSI_OVERBOUGHT, color='#e74c3c', linestyle='--', 
                label=f'Overbought ({RSI_OVERBOUGHT})')
    ax2.axhline(y=RSI_OVERSOLD, color='#2ecc71', linestyle='--', 
                label=f'Oversold ({RSI_OVERSOLD})')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax2.fill_between(plot_df.index, RSI_OVERBOUGHT, 100, color='#e74c3c', alpha=0.2)
    ax2.fill_between(plot_df.index, 0, RSI_OVERSOLD, color='#2ecc71', alpha=0.2)
    ax2.set_title('RSI', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('indicators.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved: indicators.png")


def plot_backtest_results(results, df):
    """
    Plot backtest performance.
    
    Args:
        results: Backtest results dictionary
        df: Original price DataFrame
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    
    equity_df = results['equity_curve']
    trades_df = results['trades']
    
    # 1. Price with trade markers
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='SOL Price', color='#2c3e50', linewidth=1)
    
    if len(trades_df) > 0:
        # Buy markers
        buys = trades_df[trades_df['type'] == 'BUY']
        ax1.scatter(buys['timestamp'], buys['price'], marker='^', 
                   color='#2ecc71', s=100, label='Buy', zorder=5)
        
        # Sell markers (different colors for different exit types)
        for exit_type, color in [('SELL', '#3498db'), 
                                  ('STOP_LOSS', '#e74c3c'),
                                  ('TAKE_PROFIT', '#27ae60')]:
            exits = trades_df[trades_df['type'] == exit_type]
            if len(exits) > 0:
                ax1.scatter(exits['timestamp'], exits['price'], marker='v',
                           color=color, s=100, label=exit_type.replace('_', ' ').title(), 
                           zorder=5)
    
    ax1.set_title('Trading Signals', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Equity curve
    ax2 = axes[1]
    ax2.plot(equity_df['timestamp'], equity_df['equity'], 
             color='#3498db', linewidth=1.5)
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
    
    # 3. Drawdown
    ax3 = axes[2]
    ax3.fill_between(equity_df['timestamp'], 0, equity_df['drawdown'],
                     color='#e74c3c', alpha=0.5)
    ax3.axhline(y=results['max_drawdown'], color='#c0392b', linestyle='--',
               label=f'Max DD: {results["max_drawdown"]:.1f}%')
    ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_xlabel('Date')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved: backtest_results.png")


# =============================================================================
# PARAMETER OPTIMIZATION
# =============================================================================

def optimize_parameters(df):
    """
    Grid search for optimal parameters.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        DataFrame with optimization results
    """
    results_list = []
    
    # Parameter ranges
    sma_short_range = [5, 10, 15, 20]
    sma_long_range = [20, 30, 40, 50]
    stop_loss_range = [0.05, 0.07, 0.10]
    take_profit_range = [0.10, 0.15, 0.20]
    
    total = len(sma_short_range) * len(sma_long_range) * \
            len(stop_loss_range) * len(take_profit_range)
    print(f"Testing {total} parameter combinations...")
    
    count = 0
    for sma_s in sma_short_range:
        for sma_l in sma_long_range:
            if sma_s >= sma_l:  # Short must be less than long
                continue
            
            # Recalculate indicators
            test_df = df.copy()
            test_df['sma_short'] = calculate_sma(test_df['close'], sma_s)
            test_df['sma_long'] = calculate_sma(test_df['close'], sma_l)
            test_df['rsi'] = calculate_rsi(test_df['close'], 14)
            
            test_signals = generate_signals(test_df)
            
            for sl in stop_loss_range:
                for tp in take_profit_range:
                    backtester = Backtester(
                        initial_capital=INITIAL_CAPITAL,
                        trade_pct=TRADE_PCT,
                        stop_loss=sl,
                        take_profit=tp,
                        commission=COMMISSION
                    )
                    
                    result = backtester.run(test_df, test_signals)
                    
                    results_list.append({
                        'sma_short': sma_s,
                        'sma_long': sma_l,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'total_return': result['total_return'],
                        'win_rate': result['win_rate'],
                        'max_drawdown': result['max_drawdown'],
                        'total_trades': result['total_trades']
                    })
                    count += 1
    
    print(f"Completed {count} tests")
    return pd.DataFrame(results_list)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def print_results(results):
    """Print formatted backtest results."""
    print("=" * 60)
    print("BACKTEST RESULTS")
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
    print("=" * 60)


def print_trades(trades_df):
    """Print trade history."""
    if len(trades_df) == 0:
        print("No trades executed")
        return
    
    print("\nTRADE LOG")
    print("-" * 80)
    
    for _, trade in trades_df.iterrows():
        ts = trade['timestamp'].strftime('%Y-%m-%d %H:%M')
        t_type = trade['type']
        price = trade['price']
        pnl = trade['pnl']
        pnl_pct = trade['pnl_pct']
        
        if t_type == 'BUY':
            print(f"  {ts} | BUY  @ ${price:.2f}")
        else:
            emoji = "+" if pnl > 0 else ""
            print(f"  {ts} | {t_type:11} @ ${price:.2f} | P&L: {emoji}${pnl:.2f} ({emoji}{pnl_pct:.1f}%)")
    
    # Exit type summary
    print("\nExit Type Summary:")
    exit_types = trades_df[trades_df['type'] != 'BUY']['type'].value_counts()
    for exit_type, count in exit_types.items():
        print(f"  {exit_type}: {count}")


def export_best_params(optimization_df):
    """Export best parameters as bot.js config."""
    best = optimization_df.loc[optimization_df['total_return'].idxmax()]
    
    print("\n" + "=" * 60)
    print("BEST PARAMETERS (copy to bot.js)")
    print("=" * 60)
    print(f"""
// Strategy parameters (optimized)
const SMA_SHORT_PERIOD = {int(best['sma_short'])};
const SMA_LONG_PERIOD = {int(best['sma_long'])};

const RSI_PERIOD = 14;
const RSI_OVERSOLD = 30;
const RSI_OVERBOUGHT = 70;

const STOP_LOSS = {best['stop_loss']};
const TAKE_PROFIT = {best['take_profit']};
const TRADE_PERCENTAGE = 0.1;

// Expected performance (backtest):
// Return: {best['total_return']:.2f}%
// Win Rate: {best['win_rate']:.1f}%
// Max Drawdown: {best['max_drawdown']:.2f}%
""")


def main():
    """Main execution function."""
    print("=" * 60)
    print("STRATEGY A BACKTEST")
    print("SMA + RSI + Stop Loss / Take Profit")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    if df is None:
        print("ERROR: Could not load data")
        return
    print(f"  Data range: {df.index.min()} to {df.index.max()}")
    print(f"  Total candles: {len(df)}")
    
    # Step 2: Calculate indicators
    print("\n[2/5] Calculating indicators...")
    df = add_indicators(df)
    print(f"  SMA Short ({SMA_SHORT}): {df['sma_short'].iloc[-1]:.2f}")
    print(f"  SMA Long ({SMA_LONG}): {df['sma_long'].iloc[-1]:.2f}")
    print(f"  RSI: {df['rsi'].iloc[-1]:.2f}")
    
    # Step 3: Generate signals
    print("\n[3/5] Generating signals...")
    signals = generate_signals(df)
    buy_count = (signals['signal'] == 1).sum()
    sell_count = (signals['signal'] == -1).sum()
    print(f"  Buy signals: {buy_count}")
    print(f"  Sell signals: {sell_count}")
    
    # Step 4: Run backtest
    print("\n[4/5] Running backtest...")
    backtester = Backtester(
        initial_capital=INITIAL_CAPITAL,
        trade_pct=TRADE_PCT,
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT,
        commission=COMMISSION
    )
    results = backtester.run(df, signals)
    print_results(results)
    print_trades(results['trades'])
    
    # Step 5: Visualize
    print("\n[5/5] Generating charts...")
    plot_indicators(df)
    plot_backtest_results(results, df)
    
    # Optional: Parameter optimization
    run_optimization = input("\nRun parameter optimization? (y/n): ").lower().strip()
    if run_optimization == 'y':
        print("\nOptimizing parameters...")
        opt_results = optimize_parameters(df)
        
        print("\nTop 5 by Return:")
        top5 = opt_results.nlargest(5, 'total_return')
        print(top5[['sma_short', 'sma_long', 'stop_loss', 'take_profit',
                    'total_return', 'win_rate', 'max_drawdown']].to_string(index=False))
        
        export_best_params(opt_results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
