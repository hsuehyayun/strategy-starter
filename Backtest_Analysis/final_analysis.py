"""
Final Strategy Analysis for Report
==================================
This script:
1. Runs all three strategies with updated data
2. Compares performance across different market regimes
3. Generates summary statistics for the report
4. Provides insights on when each strategy works best

Output: Tables and analysis ready for copy-paste into report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Strategy parameters (from optimization)
SMA_SHORT = 10
SMA_LONG = 50
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
STOP_LOSS = 0.05
TAKE_PROFIT = 0.15
TRADE_PCT = 0.10
COMMISSION = 0.001
INITIAL_CAPITAL = 10000

# Strategy B: F&G thresholds
FG_ENTRY_MAX = 75
FG_EXIT_EXTREME = 80

# Strategy C: Simplified SMCI
SMCI_ENTRY_MIN = 45
SMCI_EXIT_THRESHOLD = 35
WEIGHT_FG = 0.6
WEIGHT_PM = 0.4

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and prepare all data."""
    print("Loading data...")
    
    # Load price data
    df = pd.read_csv('SOL_1h_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"  Price data: {len(df)} rows")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Fetch Fear & Greed
    print("  Fetching Fear & Greed...")
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, params={'limit': 0}, timeout=30)
        fg_data = response.json()['data']
        
        fg_records = []
        for item in fg_data:
            fg_records.append({
                'date': datetime.fromtimestamp(int(item['timestamp'])),
                'fear_greed': int(item['value'])
            })
        fg_df = pd.DataFrame(fg_records).set_index('date').sort_index()
        
        # Merge with price data
        df['date'] = df.index.date
        fg_df.index = fg_df.index.date
        df['fear_greed'] = df['date'].map(lambda d: fg_df.loc[d, 'fear_greed'] 
                                          if d in fg_df.index else np.nan)
        df['fear_greed'] = df['fear_greed'].ffill().bfill().fillna(50)
        df.drop('date', axis=1, inplace=True)
        print(f"  Fear & Greed loaded")
    except:
        df['fear_greed'] = 50
        print("  Warning: Using placeholder F&G data")
    
    return df


def add_indicators(df):
    """Add all technical indicators."""
    df = df.copy()
    
    # SMA
    df['sma_short'] = df['close'].rolling(window=SMA_SHORT).mean()
    df['sma_long'] = df['close'].rolling(window=SMA_LONG).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(com=RSI_PERIOD-1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD-1, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # F&G Signal (inverted)
    df['fg_signal'] = 100 - df['fear_greed']
    
    # Polymarket Proxy
    roc = ((df['close'] - df['close'].shift(14)) / df['close'].shift(14)) * 100
    roc_score = (50 + (roc / 30) * 50).clip(0, 100)
    
    delta2 = df['close'].diff()
    gain2 = delta2.where(delta2 > 0, 0)
    loss2 = (-delta2).where(delta2 < 0, 0)
    avg_gain2 = gain2.ewm(com=13, min_periods=14).mean()
    avg_loss2 = loss2.ewm(com=13, min_periods=14).mean()
    rs2 = avg_gain2 / avg_loss2
    mrsi = 100 - (100 / (1 + rs2))
    df['polymarket_proxy'] = (0.5 * roc_score + 0.5 * mrsi).ewm(span=7).mean()
    
    # Simplified SMCI
    df['smci'] = (WEIGHT_FG * df['fg_signal'] + WEIGHT_PM * df['polymarket_proxy']).clip(0, 100).ewm(span=24).mean()
    
    # Market Regime
    daily = df['close'].resample('D').last()
    ma_50 = daily.rolling(50).mean()
    ma_200 = daily.rolling(200).mean()
    
    regime = pd.Series(index=daily.index, dtype='object')
    regime[:] = 'NEUTRAL'
    regime[(daily > ma_200) & (ma_50 > ma_200)] = 'BULL'
    regime[(daily < ma_200) & (ma_50 < ma_200)] = 'BEAR'
    
    # Map to hourly
    df['date'] = df.index.date
    regime.index = regime.index.date
    df['regime'] = df['date'].map(lambda d: regime.loc[d] if d in regime.index else 'NEUTRAL')
    df.drop('date', axis=1, inplace=True)
    
    return df.dropna()


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest_strategy(df, strategy_name):
    """
    Backtest a strategy and return results.
    """
    df = df.copy()
    
    # Generate signals based on strategy
    sma_bullish = (df['sma_short'] > df['sma_long']).astype(int)
    sma_cross_up = sma_bullish.diff() == 1
    sma_cross_down = sma_bullish.diff() == -1
    
    rsi_ok = (df['rsi'] > RSI_OVERSOLD) & (df['rsi'] < RSI_OVERBOUGHT)
    rsi_overbought = df['rsi'] > 75
    
    if strategy_name == 'A':
        # Strategy A: SMA + RSI only
        buy_signal = sma_cross_up & rsi_ok
        sell_signal = sma_cross_down | rsi_overbought
        
    elif strategy_name == 'B':
        # Strategy B: + Fear & Greed
        fg_ok = df['fear_greed'] < FG_ENTRY_MAX
        fg_extreme = df['fear_greed'] > FG_EXIT_EXTREME
        buy_signal = sma_cross_up & rsi_ok & fg_ok
        sell_signal = sma_cross_down | rsi_overbought | fg_extreme
        
    elif strategy_name == 'C':
        # Strategy C: + Simplified SMCI
        smci_bullish = df['smci'] > SMCI_ENTRY_MIN
        smci_bearish = df['smci'] < SMCI_EXIT_THRESHOLD
        buy_signal = sma_cross_up & rsi_ok & smci_bullish
        sell_signal = sma_cross_down | rsi_overbought | smci_bearish
    
    # Simulate trading
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    equity_history = []
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        regime = df['regime'].iloc[i]
        
        # Check stop loss / take profit
        if position > 0:
            pnl_pct = (price - entry_price) / entry_price
            
            if pnl_pct <= -STOP_LOSS:
                capital += position * price * (1 - COMMISSION)
                trades.append({'pnl_pct': pnl_pct, 'regime': regime, 'type': 'stop_loss'})
                position = 0
                
            elif pnl_pct >= TAKE_PROFIT:
                capital += position * price * (1 - COMMISSION)
                trades.append({'pnl_pct': pnl_pct, 'regime': regime, 'type': 'take_profit'})
                position = 0
        
        # Process signals
        if buy_signal.iloc[i] and position == 0:
            trade_amount = capital * TRADE_PCT
            position = (trade_amount * (1 - COMMISSION)) / price
            capital -= trade_amount
            entry_price = price
            entry_regime = regime
            
        elif sell_signal.iloc[i] and position > 0:
            pnl_pct = (price - entry_price) / entry_price
            capital += position * price * (1 - COMMISSION)
            trades.append({'pnl_pct': pnl_pct, 'regime': entry_regime, 'type': 'signal'})
            position = 0
        
        equity_history.append(capital + position * price)
    
    # Calculate metrics
    final_equity = capital + position * df['close'].iloc[-1]
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    if len(trades_df) > 0:
        wins = trades_df[trades_df['pnl_pct'] > 0]
        win_rate = len(wins) / len(trades_df) * 100
        avg_win = wins['pnl_pct'].mean() * 100 if len(wins) > 0 else 0
        avg_loss = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean() * 100
        avg_loss = avg_loss if not np.isnan(avg_loss) else 0
    else:
        win_rate = avg_win = avg_loss = 0
    
    equity_series = pd.Series(equity_history)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    risk_adj = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'strategy': strategy_name,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'num_trades': len(trades_df),
        'risk_adj_return': risk_adj,
        'trades_df': trades_df,
        'equity_history': equity_history
    }


# =============================================================================
# REGIME ANALYSIS
# =============================================================================

def analyze_by_regime(results_dict):
    """Analyze strategy performance by market regime."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BY MARKET REGIME")
    print("=" * 70)
    
    for strategy, results in results_dict.items():
        trades_df = results['trades_df']
        
        if len(trades_df) == 0:
            continue
        
        print(f"\n{strategy}:")
        
        for regime in ['BULL', 'BEAR', 'NEUTRAL']:
            regime_trades = trades_df[trades_df['regime'] == regime]
            
            if len(regime_trades) > 0:
                win_rate = (regime_trades['pnl_pct'] > 0).mean() * 100
                avg_return = regime_trades['pnl_pct'].mean() * 100
                print(f"  {regime:8}: {len(regime_trades):3} trades, Win Rate: {win_rate:5.1f}%, Avg Return: {avg_return:+.2f}%")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report_tables(results_dict, df):
    """Generate formatted tables for the report."""
    
    print("\n" + "=" * 70)
    print("📊 REPORT TABLES (Copy-paste ready)")
    print("=" * 70)
    
    # Table 1: Strategy Comparison
    print("\n### Table 1: Strategy Performance Comparison")
    print("| Metric | Strategy A | Strategy B | Strategy C |")
    print("|--------|------------|------------|------------|")
    
    a, b, c = results_dict['Strategy A'], results_dict['Strategy B'], results_dict['Strategy C']
    
    print(f"| Total Return | {a['total_return']:.2f}% | {b['total_return']:.2f}% | {c['total_return']:.2f}% |")
    print(f"| Max Drawdown | {a['max_drawdown']:.2f}% | {b['max_drawdown']:.2f}% | {c['max_drawdown']:.2f}% |")
    print(f"| Win Rate | {a['win_rate']:.1f}% | {b['win_rate']:.1f}% | {c['win_rate']:.1f}% |")
    print(f"| Total Trades | {a['num_trades']} | {b['num_trades']} | {c['num_trades']} |")
    print(f"| Risk-Adj Return | {a['risk_adj_return']:.2f} | {b['risk_adj_return']:.2f} | {c['risk_adj_return']:.2f} |")
    
    # Table 2: Strategy Descriptions
    print("\n### Table 2: Strategy Descriptions")
    print("| Strategy | Entry Conditions | Exit Conditions | Key Feature |")
    print("|----------|------------------|-----------------|-------------|")
    print("| A | SMA Cross + RSI 30-70 | SMA Cross + RSI>75 | Technical only |")
    print("| B | A + F&G < 75 | A + F&G > 80 | + Sentiment filter |")
    print("| C | A + SMCI > 45 | A + SMCI < 35 | + Composite index |")
    
    # Table 3: Data Summary
    print("\n### Table 3: Data Summary")
    daily = df['close'].resample('D').last().dropna()
    returns = daily.pct_change().dropna()
    
    print(f"| Item | Value |")
    print(f"|------|-------|")
    print(f"| Date Range | {df.index.min().date()} to {df.index.max().date()} |")
    print(f"| Total Candles | {len(df):,} |")
    print(f"| Total Days | {len(daily):,} |")
    print(f"| Price Range | ${df['close'].min():.2f} - ${df['close'].max():.2f} |")
    print(f"| Total Return (Buy & Hold) | {(daily.iloc[-1]/daily.iloc[0]-1)*100:.2f}% |")
    print(f"| Annualized Volatility | {returns.std()*np.sqrt(365)*100:.1f}% |")
    print(f"| Sharpe Ratio | {returns.mean()/returns.std()*np.sqrt(365):.2f} |")
    
    # Current Market State
    print("\n### Table 4: Current Market State")
    print(f"| Indicator | Value | Interpretation |")
    print(f"|-----------|-------|----------------|")
    print(f"| Current Price | ${df['close'].iloc[-1]:.2f} | - |")
    print(f"| Fear & Greed | {df['fear_greed'].iloc[-1]:.0f} | {'Extreme Fear' if df['fear_greed'].iloc[-1] < 25 else 'Fear' if df['fear_greed'].iloc[-1] < 45 else 'Neutral' if df['fear_greed'].iloc[-1] < 55 else 'Greed' if df['fear_greed'].iloc[-1] < 75 else 'Extreme Greed'} |")
    print(f"| SMCI | {df['smci'].iloc[-1]:.1f} | {'Bullish' if df['smci'].iloc[-1] > 55 else 'Bearish' if df['smci'].iloc[-1] < 45 else 'Neutral'} |")
    print(f"| Market Regime | {df['regime'].iloc[-1]} | Based on MA50/MA200 |")


def generate_insights(results_dict):
    """Generate key insights for the report."""
    print("\n" + "=" * 70)
    print("💡 KEY INSIGHTS FOR REPORT")
    print("=" * 70)
    
    a = results_dict['Strategy A']
    b = results_dict['Strategy B']
    c = results_dict['Strategy C']
    
    # Find winner
    returns = {'A': a['total_return'], 'B': b['total_return'], 'C': c['total_return']}
    winner = max(returns, key=returns.get)
    
    risk_adj = {'A': a['risk_adj_return'], 'B': b['risk_adj_return'], 'C': c['risk_adj_return']}
    risk_winner = max(risk_adj, key=risk_adj.get)
    
    print(f"""
1. PERFORMANCE WINNER: Strategy {winner}
   - Highest return: {returns[winner]:.2f}%
   - Adding sentiment indicators {'improved' if winner != 'A' else 'did not improve'} performance

2. RISK-ADJUSTED WINNER: Strategy {risk_winner}
   - Highest Return/MaxDD ratio: {risk_adj[risk_winner]:.2f}
   - Better risk management through {'sentiment filters' if risk_winner != 'A' else 'technical signals'}

3. TRADE FREQUENCY:
   - Strategy A: {a['num_trades']} trades (baseline)
   - Strategy B: {b['num_trades']} trades ({(b['num_trades']-a['num_trades'])/a['num_trades']*100:+.1f}% vs A)
   - Strategy C: {c['num_trades']} trades ({(c['num_trades']-a['num_trades'])/a['num_trades']*100:+.1f}% vs A)
   - Fewer trades = more selective entry, potentially better quality

4. FEAR & GREED EFFECTIVENESS:
   - Strategy B adds F&G filter
   - Impact: {'Positive' if b['total_return'] > a['total_return'] else 'Negative'} ({b['total_return']-a['total_return']:+.2f}% difference)

5. SMCI CONTRIBUTION:
   - Strategy C uses composite index (F&G + Momentum)
   - More aggressive filtering leads to fewer trades
   - Trade-off: Lower return but potentially better risk control

6. RECOMMENDATION FOR COMPETITION:
   - Primary: Strategy {winner} (highest return)
   - If market turns volatile: Consider Strategy {risk_winner} (best risk-adjusted)
""")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(results_dict, df):
    """Create comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Equity Curves
    ax1 = axes[0, 0]
    for name, results in results_dict.items():
        ax1.plot(results['equity_history'], label=f"{name} ({results['total_return']:.1f}%)", linewidth=1.5)
    ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Equity Curves Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Value (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Return Comparison Bar
    ax2 = axes[0, 1]
    strategies = list(results_dict.keys())
    returns = [results_dict[s]['total_return'] for s in strategies]
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    bars = ax2.bar(strategies, returns, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_title('Total Return Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Return (%)')
    for bar, ret in zip(bars, returns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{ret:.1f}%', ha='center', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Risk Metrics
    ax3 = axes[1, 0]
    x = np.arange(len(strategies))
    width = 0.35
    
    max_dd = [abs(results_dict[s]['max_drawdown']) for s in strategies]
    risk_adj = [results_dict[s]['risk_adj_return'] for s in strategies]
    
    ax3.bar(x - width/2, max_dd, width, label='Max Drawdown (%)', color='#e74c3c', alpha=0.7)
    ax3.bar(x + width/2, risk_adj, width, label='Risk-Adj Return', color='#27ae60', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies)
    ax3.set_title('Risk Metrics', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Trade Statistics
    ax4 = axes[1, 1]
    num_trades = [results_dict[s]['num_trades'] for s in strategies]
    win_rates = [results_dict[s]['win_rate'] for s in strategies]
    
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar(x - width/2, num_trades, width, label='Trades', color='#3498db', alpha=0.7)
    bars2 = ax4_twin.bar(x + width/2, win_rates, width, label='Win Rate', color='#f39c12', alpha=0.7)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies)
    ax4.set_ylabel('Number of Trades', color='#3498db')
    ax4_twin.set_ylabel('Win Rate (%)', color='#f39c12')
    ax4.set_title('Trade Statistics', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nChart saved: strategy_comparison.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("FINAL STRATEGY ANALYSIS")
    print("=" * 70)
    
    # Load and prepare data
    df = load_data()
    df = add_indicators(df)
    
    # Run all strategies
    print("\nRunning backtests...")
    results_dict = {}
    
    for strategy in ['A', 'B', 'C']:
        print(f"  Strategy {strategy}...")
        results = backtest_strategy(df, strategy)
        results_dict[f'Strategy {strategy}'] = results
    
    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    for name, results in results_dict.items():
        print(f"\n{name}:")
        print(f"  Return: {results['total_return']:.2f}%")
        print(f"  Max DD: {results['max_drawdown']:.2f}%")
        print(f"  Win Rate: {results['win_rate']:.1f}%")
        print(f"  Trades: {results['num_trades']}")
        print(f"  Risk-Adj: {results['risk_adj_return']:.2f}")
    
    # Analyze by regime
    analyze_by_regime(results_dict)
    
    # Generate report tables
    generate_report_tables(results_dict, df)
    
    # Generate insights
    generate_insights(results_dict)
    
    # Plot comparison
    print("\nGenerating comparison chart...")
    try:
        plot_comparison(results_dict, df)
    except Exception as e:
        print(f"  Could not generate chart: {e}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nReady for report writing!")
    
    return results_dict, df


if __name__ == "__main__":
    results, df = main()
