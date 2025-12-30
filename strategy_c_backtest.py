"""
Strategy C Backtest: SMA + RSI + SMCI (Smart Money Composite Index)
====================================================================
Competition: Trading Bot Competition (Jan 1-14, 2025)
Asset: SOL (Solana)

This strategy combines:
1. Technical indicators (SMA crossover, RSI)
2. Smart Money Composite Index (SMCI) - a novel sentiment composite

SMCI Components:
- Fear & Greed Index (40%): Retail sentiment (contrarian)
- Polymarket Proxy (30%): Market expectations from momentum
- Funding Rate (30%): Leveraged trader sentiment (contrarian)

Strategy Rules:
---------------
ENTRY (all conditions must be met):
  1. SMA short > SMA long (golden cross confirmed)
  2. RSI between 30-70 (not overbought/oversold)
  3. SMCI > 50 (smart money sentiment positive)

EXIT (any condition triggers):
  1. Stop loss: -5%
  2. Take profit: +15%
  3. SMA death cross
  4. RSI > 75 (overbought)
  5. SMCI < 35 (smart money sentiment turns negative)

Academic Foundation:
-------------------
- Wisdom of Crowds (Surowiecki, 2004): Prediction markets aggregate information
- Behavioral Finance: Contrarian indicators (F&G, Funding Rate)
- Technical Analysis: Trend following (SMA crossover)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURATION (Optimized from Strategy A/B)
# =============================================================================

# Technical indicator parameters
SMA_SHORT = 10          # Short-term SMA period
SMA_LONG = 50           # Long-term SMA period (optimized)
RSI_PERIOD = 14         # RSI calculation period
RSI_OVERSOLD = 30       # RSI oversold threshold
RSI_OVERBOUGHT = 70     # RSI overbought threshold

# SMCI thresholds [NEW - Strategy C]
SMCI_ENTRY_MIN = 50     # Only enter when SMCI above this
SMCI_EXIT_THRESHOLD = 35 # Exit when SMCI falls below this

# Risk management (optimized)
STOP_LOSS = 0.05        # Stop loss at -5%
TAKE_PROFIT = 0.15      # Take profit at +15%
TRADE_PCT = 0.10        # Use 10% of capital per trade
COMMISSION = 0.001      # 0.1% commission per trade

# Backtest settings
INITIAL_CAPITAL = 10000  # Starting capital in USD

# =============================================================================
# DATA LOADING
# =============================================================================

def load_price_data(filepath='SOL_1h_data.csv'):
    """Load historical price data from CSV."""
    print(f"Loading price data from {filepath}...")
    
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
    
    print(f"  Loaded {len(df)} candles")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def fetch_fear_greed_history(limit=0):
    """Fetch Fear & Greed Index history."""
    print("Fetching Fear & Greed Index...")
    
    url = "https://api.alternative.me/fng/"
    params = {'limit': limit, 'format': 'json'}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if 'data' not in data:
            return None
        
        records = []
        for item in data['data']:
            records.append({
                'date': datetime.fromtimestamp(int(item['timestamp'])),
                'fear_greed': int(item['value'])
            })
        
        fg_df = pd.DataFrame(records)
        fg_df.set_index('date', inplace=True)
        fg_df.sort_index(inplace=True)
        
        print(f"  Fetched {len(fg_df)} days of Fear & Greed data")
        return fg_df
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_sma(data, period):
    """Calculate Simple Moving Average."""
    return data.rolling(window=period).mean()


def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def add_technical_indicators(df):
    """Add all technical indicators to the DataFrame."""
    df = df.copy()
    
    df['sma_short'] = calculate_sma(df['close'], SMA_SHORT)
    df['sma_long'] = calculate_sma(df['close'], SMA_LONG)
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    
    return df


# =============================================================================
# SMCI CALCULATION (Simplified for backtest)
# =============================================================================

def calculate_momentum_proxy(prices, period=14):
    """
    Calculate Polymarket proxy from price momentum.
    Returns 0-100 score.
    """
    # Rate of change
    roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
    
    # Normalize to 0-100
    roc_score = (50 + (roc / 30) * 50).clip(0, 100)
    
    # Simple momentum RSI
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    mrsi = 100 - (100 / (1 + rs))
    
    # Combine
    proxy = 0.5 * roc_score + 0.5 * mrsi
    
    return proxy.ewm(span=7).mean()


def calculate_funding_signal_from_price(prices, period=14):
    """
    Estimate funding signal from price volatility.
    
    In reality, we'd use actual funding rates. For backtest without
    funding data, we estimate based on price behavior:
    - Large up moves → likely positive funding → bearish signal (contrarian)
    - Large down moves → likely negative funding → bullish signal (contrarian)
    
    Returns 0-100 score (higher = more bullish).
    """
    # Calculate recent volatility direction
    returns = prices.pct_change(period)
    
    # Contrarian: large positive returns → lower score
    # Large negative returns → higher score
    signal = 50 - (returns * 500)  # Scale factor
    
    return signal.clip(0, 100).ewm(span=7).mean()


def calculate_smci(fear_greed_signal, polymarket_proxy, funding_signal,
                   w_fg=0.4, w_pm=0.3, w_fr=0.3):
    """
    Calculate Smart Money Composite Index.
    
    All inputs should be 0-100 where higher = more bullish.
    """
    smci = w_fg * fear_greed_signal + w_pm * polymarket_proxy + w_fr * funding_signal
    return smci.clip(0, 100)


def add_smci_to_dataframe(df, fear_greed_df):
    """
    Add SMCI components and final score to price DataFrame.
    """
    df = df.copy()
    
    # 1. Merge Fear & Greed (daily to hourly)
    df['date'] = df.index.date
    
    if fear_greed_df is not None:
        fg_daily = fear_greed_df[['fear_greed']].copy()
        fg_daily.index = fg_daily.index.date
        df['fear_greed'] = df['date'].map(lambda d: fg_daily.loc[d, 'fear_greed'] 
                                           if d in fg_daily.index else np.nan)
        df['fear_greed'] = df['fear_greed'].ffill().bfill().fillna(50)
    else:
        df['fear_greed'] = 50
    
    # Fear & Greed Signal (inverted - contrarian)
    df['fg_signal'] = 100 - df['fear_greed']
    
    # 2. Calculate Polymarket Proxy
    df['polymarket_proxy'] = calculate_momentum_proxy(df['close'])
    
    # 3. Calculate Funding Signal (estimated)
    df['funding_signal'] = calculate_funding_signal_from_price(df['close'])
    
    # 4. Calculate SMCI
    df['smci'] = calculate_smci(
        df['fg_signal'],
        df['polymarket_proxy'],
        df['funding_signal']
    )
    
    # 5. Smooth SMCI
    df['smci_smoothed'] = df['smci'].ewm(span=24).mean()  # 24-hour smoothing
    
    # Clean up
    df.drop('date', axis=1, inplace=True)
    
    return df


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(df):
    """
    Generate buy/sell signals based on Strategy C rules.
    
    Buy signal (1):
        - SMA short crosses above SMA long (golden cross)
        - RSI is between 30-70
        - SMCI > 50 (smart money sentiment positive) [NEW]
        
    Sell signal (-1):
        - SMA short crosses below SMA long (death cross)
        - OR RSI > 75 (overbought)
        - OR SMCI < 35 (smart money sentiment turns negative) [NEW]
    """
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    
    # SMA crossover detection
    sma_bullish = (df['sma_short'] > df['sma_long']).astype(int)
    sma_cross_up = sma_bullish.diff() == 1
    sma_cross_down = sma_bullish.diff() == -1
    
    # RSI conditions
    rsi_ok = (df['rsi'] > RSI_OVERSOLD) & (df['rsi'] < RSI_OVERBOUGHT)
    rsi_overbought = df['rsi'] > 75
    
    # SMCI conditions [NEW]
    smci_bullish = df['smci_smoothed'] > SMCI_ENTRY_MIN
    smci_bearish = df['smci_smoothed'] < SMCI_EXIT_THRESHOLD
    
    # Generate signals
    buy_condition = sma_cross_up & rsi_ok & smci_bullish
    signals.loc[buy_condition, 'signal'] = 1
    
    sell_condition = sma_cross_down | rsi_overbought | smci_bearish
    signals.loc[sell_condition, 'signal'] = -1
    
    return signals


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class Backtester:
    """Backtesting engine with SMCI tracking."""
    
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
            smci_value = df['smci_smoothed'].iloc[i] if 'smci_smoothed' in df.columns else 50
            
            current_equity = capital + position * price
            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'price': price,
                'position': position,
                'smci': smci_value
            })
            
            # Check stop loss / take profit
            if position > 0:
                pnl_pct = (price - entry_price) / entry_price
                
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
                        'smci': smci_value
                    })
                    position = 0
                    continue
                
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
                        'smci': smci_value
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
                    'smci': smci_value
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
                    'smci': smci_value
                })
                position = 0
        
        # Calculate metrics
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
        
        # Count SMCI-triggered exits
        smci_exits = 0
        if len(trades_df) > 0 and 'smci' in trades_df.columns:
            sell_trades = trades_df[trades_df['type'] == 'SELL']
            smci_exits = (sell_trades['smci'] < SMCI_EXIT_THRESHOLD).sum()
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'smci_triggered_exits': smci_exits,
            'trades': trades_df,
            'equity_curve': equity_df
        }


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def run_strategy_a(df):
    """Run Strategy A (SMA + RSI only) for comparison."""
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    
    sma_bullish = (df['sma_short'] > df['sma_long']).astype(int)
    sma_cross_up = sma_bullish.diff() == 1
    sma_cross_down = sma_bullish.diff() == -1
    
    rsi_ok = (df['rsi'] > RSI_OVERSOLD) & (df['rsi'] < RSI_OVERBOUGHT)
    rsi_overbought = df['rsi'] > 75
    
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


def run_strategy_b(df):
    """Run Strategy B (SMA + RSI + Fear & Greed) for comparison."""
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    
    sma_bullish = (df['sma_short'] > df['sma_long']).astype(int)
    sma_cross_up = sma_bullish.diff() == 1
    sma_cross_down = sma_bullish.diff() == -1
    
    rsi_ok = (df['rsi'] > RSI_OVERSOLD) & (df['rsi'] < RSI_OVERBOUGHT)
    rsi_overbought = df['rsi'] > 75
    
    fg_ok = df['fear_greed'] < 75
    fg_extreme = df['fear_greed'] > 80
    
    buy_condition = sma_cross_up & rsi_ok & fg_ok
    signals.loc[buy_condition, 'signal'] = 1
    
    sell_condition = sma_cross_down | rsi_overbought | fg_extreme
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
# VISUALIZATION
# =============================================================================

def plot_strategy_c_results(results, df):
    """Plot Strategy C backtest results with SMCI."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                             gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    
    equity_df = results['equity_curve']
    trades_df = results['trades']
    
    # 1. Price with trades
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
    
    ax1.set_title('Strategy C: SMA + RSI + SMCI (Smart Money Composite Index)',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Equity curve
    ax2 = axes[1]
    ax2.plot(equity_df['timestamp'], equity_df['equity'],
             color='#3498db', linewidth=1.5)
    ax2.axhline(y=results['initial_capital'], color='gray', linestyle='--')
    ax2.fill_between(equity_df['timestamp'], results['initial_capital'],
                     equity_df['equity'],
                     where=equity_df['equity'] >= results['initial_capital'],
                     color='#2ecc71', alpha=0.3)
    ax2.fill_between(equity_df['timestamp'], results['initial_capital'],
                     equity_df['equity'],
                     where=equity_df['equity'] < results['initial_capital'],
                     color='#e74c3c', alpha=0.3)
    ax2.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Portfolio Value (USD)')
    ax2.grid(True, alpha=0.3)
    
    # 3. SMCI
    ax3 = axes[2]
    smci_colors = df['smci_smoothed'].apply(
        lambda x: '#27ae60' if x >= 55 else '#e74c3c' if x < 45 else '#f1c40f'
    )
    ax3.fill_between(df.index, 0, df['smci_smoothed'], color='#9b59b6', alpha=0.5)
    ax3.axhline(y=SMCI_ENTRY_MIN, color='#27ae60', linestyle='--',
                label=f'Entry Min ({SMCI_ENTRY_MIN})')
    ax3.axhline(y=SMCI_EXIT_THRESHOLD, color='#e74c3c', linestyle='--',
                label=f'Exit Threshold ({SMCI_EXIT_THRESHOLD})')
    ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax3.set_title('SMCI - Smart Money Composite Index', fontsize=12, fontweight='bold')
    ax3.set_ylabel('SMCI Score')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. SMCI Components
    ax4 = axes[3]
    ax4.plot(df.index, df['fg_signal'], label='Fear & Greed (40%)',
             color='#3498db', alpha=0.7, linewidth=1)
    ax4.plot(df.index, df['polymarket_proxy'], label='Polymarket Proxy (30%)',
             color='#9b59b6', alpha=0.7, linewidth=1)
    ax4.plot(df.index, df['funding_signal'], label='Funding Signal (30%)',
             color='#e67e22', alpha=0.7, linewidth=1)
    ax4.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax4.set_title('SMCI Components', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Score')
    ax4.set_xlabel('Date')
    ax4.set_ylim(0, 100)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('strategy_c_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved: strategy_c_results.png")


def print_results(results, strategy_name):
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
    
    if 'smci_triggered_exits' in results:
        print(f"\nSMCI Impact:")
        print(f"  SMCI Triggered Exits: {results['smci_triggered_exits']}")


def print_comparison(results_a, results_b, results_c):
    """Print comparison of all three strategies."""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON: A vs B vs C")
    print("=" * 80)
    print(f"{'Metric':<25} {'Strategy A':>15} {'Strategy B':>15} {'Strategy C':>15}")
    print("-" * 80)
    print(f"{'Total Return':<25} {results_a['total_return']:>14.2f}% {results_b['total_return']:>14.2f}% {results_c['total_return']:>14.2f}%")
    print(f"{'Win Rate':<25} {results_a['win_rate']:>14.1f}% {results_b['win_rate']:>14.1f}% {results_c['win_rate']:>14.1f}%")
    print(f"{'Max Drawdown':<25} {results_a['max_drawdown']:>14.2f}% {results_b['max_drawdown']:>14.2f}% {results_c['max_drawdown']:>14.2f}%")
    print(f"{'Total Trades':<25} {results_a['total_trades']:>15} {results_b['total_trades']:>15} {results_c['total_trades']:>15}")
    print(f"{'Avg Win':<25} {results_a['avg_win']:>14.2f}% {results_b['avg_win']:>14.2f}% {results_c['avg_win']:>14.2f}%")
    print(f"{'Avg Loss':<25} {results_a['avg_loss']:>14.2f}% {results_b['avg_loss']:>14.2f}% {results_c['avg_loss']:>14.2f}%")
    print("=" * 80)
    
    # Determine winner
    returns = {'A': results_a['total_return'], 'B': results_b['total_return'], 'C': results_c['total_return']}
    winner = max(returns, key=returns.get)
    print(f"\n🏆 WINNER: Strategy {winner} with {returns[winner]:.2f}% return!")
    
    # Risk-adjusted comparison
    sharpe_proxy_a = results_a['total_return'] / abs(results_a['max_drawdown']) if results_a['max_drawdown'] != 0 else 0
    sharpe_proxy_b = results_b['total_return'] / abs(results_b['max_drawdown']) if results_b['max_drawdown'] != 0 else 0
    sharpe_proxy_c = results_c['total_return'] / abs(results_c['max_drawdown']) if results_c['max_drawdown'] != 0 else 0
    
    print(f"\nRisk-Adjusted Return (Return/MaxDD):")
    print(f"  Strategy A: {sharpe_proxy_a:.2f}")
    print(f"  Strategy B: {sharpe_proxy_b:.2f}")
    print(f"  Strategy C: {sharpe_proxy_c:.2f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("STRATEGY C BACKTEST")
    print("SMA + RSI + SMCI (Smart Money Composite Index)")
    print("=" * 60)
    
    # Step 1: Load price data
    print("\n[1/7] Loading price data...")
    try:
        df = load_price_data('SOL_1h_data.csv')
    except:
        print("ERROR: SOL_1h_data.csv not found!")
        print("Please place the file in the current directory.")
        return
    
    # Step 2: Fetch Fear & Greed data
    print("\n[2/7] Fetching Fear & Greed Index...")
    fg_df = fetch_fear_greed_history(limit=0)
    
    # Step 3: Add technical indicators
    print("\n[3/7] Calculating technical indicators...")
    df = add_technical_indicators(df)
    
    # Step 4: Add SMCI
    print("\n[4/7] Calculating SMCI...")
    df = add_smci_to_dataframe(df, fg_df)
    
    print(f"  Latest SMCI: {df['smci_smoothed'].iloc[-1]:.1f}")
    print(f"  Latest F&G Signal: {df['fg_signal'].iloc[-1]:.1f}")
    print(f"  Latest Polymarket Proxy: {df['polymarket_proxy'].iloc[-1]:.1f}")
    print(f"  Latest Funding Signal: {df['funding_signal'].iloc[-1]:.1f}")
    
    # Step 5: Generate signals and run backtest
    print("\n[5/7] Running Strategy C backtest...")
    signals_c = generate_signals(df)
    
    backtester = Backtester(
        initial_capital=INITIAL_CAPITAL,
        trade_pct=TRADE_PCT,
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT,
        commission=COMMISSION
    )
    
    results_c = backtester.run(df, signals_c)
    print_results(results_c, "Strategy C (SMA + RSI + SMCI)")
    
    # Step 6: Run comparison strategies
    print("\n[6/7] Running comparison strategies...")
    
    print("\nRunning Strategy A (SMA + RSI)...")
    results_a = run_strategy_a(df)
    print_results(results_a, "Strategy A (SMA + RSI)")
    
    print("\nRunning Strategy B (SMA + RSI + F&G)...")
    results_b = run_strategy_b(df)
    print_results(results_b, "Strategy B (SMA + RSI + F&G)")
    
    # Print comparison
    print_comparison(results_a, results_b, results_c)
    
    # Step 7: Visualize
    print("\n[7/7] Generating charts...")
    try:
        plot_strategy_c_results(results_c, df)
    except Exception as e:
        print(f"  Could not generate chart: {e}")
    
    # Print final parameters
    print("\n" + "=" * 60)
    print("STRATEGY C PARAMETERS (for bot.js)")
    print("=" * 60)
    print(f"""
// Strategy C: SMA + RSI + SMCI
const SMA_SHORT_PERIOD = {SMA_SHORT};
const SMA_LONG_PERIOD = {SMA_LONG};

const RSI_PERIOD = {RSI_PERIOD};
const RSI_OVERSOLD = {RSI_OVERSOLD};
const RSI_OVERBOUGHT = {RSI_OVERBOUGHT};

// SMCI thresholds
const SMCI_ENTRY_MIN = {SMCI_ENTRY_MIN};      // Only enter when above
const SMCI_EXIT_THRESHOLD = {SMCI_EXIT_THRESHOLD};  // Exit when below

// SMCI Component Weights
const WEIGHT_FEAR_GREED = 0.4;
const WEIGHT_POLYMARKET = 0.3;
const WEIGHT_FUNDING = 0.3;

// Risk management
const STOP_LOSS = {STOP_LOSS};
const TAKE_PROFIT = {TAKE_PROFIT};
const TRADE_PERCENTAGE = {TRADE_PCT};
""")
    
    print("\nDone!")
    
    return results_c, df


if __name__ == "__main__":
    results, df = main()
