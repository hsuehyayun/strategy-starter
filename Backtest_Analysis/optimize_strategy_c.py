"""
Strategy C Parameter Optimizer
==============================
Optimizes SMCI parameters with Walk-Forward validation to avoid overfitting.

Parameters to optimize:
- SMCI_ENTRY_MIN: Minimum SMCI score to enter (default: 50)
- SMCI_EXIT_THRESHOLD: Exit when SMCI falls below (default: 35)
- WEIGHT_FEAR_GREED: F&G weight in SMCI (default: 0.4)
- WEIGHT_POLYMARKET: Polymarket proxy weight (default: 0.3)
- WEIGHT_FUNDING: Funding rate weight (default: 0.3)

Methodology:
1. Split data into Train (70%) and Test (30%)
2. Optimize on Train set
3. Validate on Test set (unseen data)
4. Report both in-sample and out-of-sample performance
"""

import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Fixed parameters (from Strategy A/B optimization)
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

# Parameter ranges to test
PARAM_GRID = {
    'smci_entry_min': [40, 45, 50, 55],
    'smci_exit_threshold': [30, 35, 40],
    'weight_fg': [0.3, 0.4, 0.5],
    'weight_pm': [0.2, 0.3, 0.4],
    # weight_fr is calculated as 1 - weight_fg - weight_pm
}

# Train/Test split ratio
TRAIN_RATIO = 0.7

# =============================================================================
# DATA LOADING & INDICATORS
# =============================================================================

def load_and_prepare_data(filepath='SOL_1h_data.csv'):
    """Load data and add all indicators."""
    print("Loading data...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add technical indicators
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
    
    # Fear & Greed (fetch from API)
    print("Fetching Fear & Greed data...")
    try:
        import requests
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
    except:
        print("  Could not fetch F&G, using placeholder")
        df['fear_greed'] = 50
    
    # F&G Signal (inverted - contrarian)
    df['fg_signal'] = 100 - df['fear_greed']
    
    # Polymarket Proxy (momentum-based)
    roc = ((df['close'] - df['close'].shift(14)) / df['close'].shift(14)) * 100
    roc_score = (50 + (roc / 30) * 50).clip(0, 100)
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    mrsi = 100 - (100 / (1 + rs))
    
    df['polymarket_proxy'] = (0.5 * roc_score + 0.5 * mrsi).ewm(span=7).mean()
    
    # Funding Signal (estimated from price)
    returns = df['close'].pct_change(14)
    df['funding_signal'] = (50 - (returns * 500)).clip(0, 100).ewm(span=7).mean()
    
    # Drop NaN rows
    df = df.dropna()
    
    print(f"  Prepared {len(df)} rows")
    return df


# =============================================================================
# SMCI CALCULATION (with variable weights)
# =============================================================================

def calculate_smci(df, weight_fg, weight_pm, weight_fr):
    """Calculate SMCI with given weights."""
    smci = (
        weight_fg * df['fg_signal'] +
        weight_pm * df['polymarket_proxy'] +
        weight_fr * df['funding_signal']
    )
    return smci.clip(0, 100).ewm(span=24).mean()


# =============================================================================
# BACKTESTER
# =============================================================================

def run_backtest(df, smci_entry_min, smci_exit_threshold, 
                 weight_fg, weight_pm, weight_fr):
    """
    Run backtest with given parameters.
    Returns performance metrics.
    """
    df = df.copy()
    
    # Calculate SMCI with given weights
    df['smci'] = calculate_smci(df, weight_fg, weight_pm, weight_fr)
    
    # Generate signals
    sma_bullish = (df['sma_short'] > df['sma_long']).astype(int)
    sma_cross_up = sma_bullish.diff() == 1
    sma_cross_down = sma_bullish.diff() == -1
    
    rsi_ok = (df['rsi'] > RSI_OVERSOLD) & (df['rsi'] < RSI_OVERBOUGHT)
    rsi_overbought = df['rsi'] > 75
    
    smci_bullish = df['smci'] > smci_entry_min
    smci_bearish = df['smci'] < smci_exit_threshold
    
    # Buy/Sell signals
    buy_signal = sma_cross_up & rsi_ok & smci_bullish
    sell_signal = sma_cross_down | rsi_overbought | smci_bearish
    
    # Simulate trading
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    equity_history = [INITIAL_CAPITAL]
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        
        # Check stop loss / take profit
        if position > 0:
            pnl_pct = (price - entry_price) / entry_price
            
            if pnl_pct <= -STOP_LOSS:
                capital += position * price * (1 - COMMISSION)
                trades.append(pnl_pct)
                position = 0
                
            elif pnl_pct >= TAKE_PROFIT:
                capital += position * price * (1 - COMMISSION)
                trades.append(pnl_pct)
                position = 0
        
        # Process signals
        if buy_signal.iloc[i] and position == 0:
            trade_amount = capital * TRADE_PCT
            position = (trade_amount * (1 - COMMISSION)) / price
            capital -= trade_amount
            entry_price = price
            
        elif sell_signal.iloc[i] and position > 0:
            pnl_pct = (price - entry_price) / entry_price
            capital += position * price * (1 - COMMISSION)
            trades.append(pnl_pct)
            position = 0
        
        equity_history.append(capital + position * price)
    
    # Calculate metrics
    final_equity = capital + position * df['close'].iloc[-1]
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # Win rate
    if trades:
        wins = [t for t in trades if t > 0]
        win_rate = len(wins) / len(trades) * 100
    else:
        win_rate = 0
    
    # Max drawdown
    equity_series = pd.Series(equity_history)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    # Risk-adjusted return
    if max_drawdown != 0:
        risk_adj_return = total_return / abs(max_drawdown)
    else:
        risk_adj_return = total_return
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades),
        'risk_adj_return': risk_adj_return
    }


# =============================================================================
# PARAMETER OPTIMIZATION
# =============================================================================

def optimize_parameters(df_train, param_grid):
    """
    Grid search for best parameters on training data.
    """
    print("\nOptimizing parameters on training data...")
    
    results = []
    
    # Generate all parameter combinations
    entry_mins = param_grid['smci_entry_min']
    exit_thresholds = param_grid['smci_exit_threshold']
    weight_fgs = param_grid['weight_fg']
    weight_pms = param_grid['weight_pm']
    
    total_combinations = len(entry_mins) * len(exit_thresholds) * len(weight_fgs) * len(weight_pms)
    print(f"  Testing {total_combinations} parameter combinations...")
    
    count = 0
    for entry_min in entry_mins:
        for exit_thresh in exit_thresholds:
            for w_fg in weight_fgs:
                for w_pm in weight_pms:
                    # Calculate funding weight (must sum to 1)
                    w_fr = 1.0 - w_fg - w_pm
                    
                    # Skip invalid combinations
                    if w_fr < 0.1 or w_fr > 0.5:
                        continue
                    
                    # Skip if exit threshold >= entry min
                    if exit_thresh >= entry_min:
                        continue
                    
                    # Run backtest
                    metrics = run_backtest(
                        df_train,
                        smci_entry_min=entry_min,
                        smci_exit_threshold=exit_thresh,
                        weight_fg=w_fg,
                        weight_pm=w_pm,
                        weight_fr=w_fr
                    )
                    
                    results.append({
                        'smci_entry_min': entry_min,
                        'smci_exit_threshold': exit_thresh,
                        'weight_fg': w_fg,
                        'weight_pm': w_pm,
                        'weight_fr': w_fr,
                        **metrics
                    })
                    
                    count += 1
                    if count % 20 == 0:
                        print(f"  Completed {count} tests...")
    
    results_df = pd.DataFrame(results)
    print(f"  Tested {len(results_df)} valid combinations")
    
    return results_df


def walk_forward_optimize(df, param_grid, train_ratio=0.7):
    """
    Walk-forward optimization:
    1. Split data into train/test
    2. Optimize on train
    3. Validate on test
    """
    # Split data
    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    print(f"\nData Split:")
    print(f"  Train: {df_train.index.min().date()} to {df_train.index.max().date()} ({len(df_train)} rows)")
    print(f"  Test:  {df_test.index.min().date()} to {df_test.index.max().date()} ({len(df_test)} rows)")
    
    # Optimize on training data
    train_results = optimize_parameters(df_train, param_grid)
    
    # Find best parameters (by risk-adjusted return)
    best_idx = train_results['risk_adj_return'].idxmax()
    best_params = train_results.loc[best_idx]
    
    print("\n" + "=" * 60)
    print("BEST PARAMETERS (Training Set)")
    print("=" * 60)
    print(f"  SMCI Entry Min:     {best_params['smci_entry_min']}")
    print(f"  SMCI Exit Threshold: {best_params['smci_exit_threshold']}")
    print(f"  Weight F&G:         {best_params['weight_fg']}")
    print(f"  Weight Polymarket:  {best_params['weight_pm']}")
    print(f"  Weight Funding:     {best_params['weight_fr']}")
    print(f"\n  Training Performance:")
    print(f"    Return:           {best_params['total_return']:.2f}%")
    print(f"    Max Drawdown:     {best_params['max_drawdown']:.2f}%")
    print(f"    Risk-Adj Return:  {best_params['risk_adj_return']:.2f}")
    print(f"    Trades:           {best_params['num_trades']:.0f}")
    
    # Validate on test data
    test_metrics = run_backtest(
        df_test,
        smci_entry_min=best_params['smci_entry_min'],
        smci_exit_threshold=best_params['smci_exit_threshold'],
        weight_fg=best_params['weight_fg'],
        weight_pm=best_params['weight_pm'],
        weight_fr=best_params['weight_fr']
    )
    
    print(f"\n  Test Performance (Out-of-Sample):")
    print(f"    Return:           {test_metrics['total_return']:.2f}%")
    print(f"    Max Drawdown:     {test_metrics['max_drawdown']:.2f}%")
    print(f"    Risk-Adj Return:  {test_metrics['risk_adj_return']:.2f}")
    print(f"    Trades:           {test_metrics['num_trades']}")
    
    # Check for overfitting
    train_return = best_params['total_return']
    test_return = test_metrics['total_return']
    
    print("\n" + "=" * 60)
    print("OVERFITTING CHECK")
    print("=" * 60)
    
    if test_return >= train_return * 0.7:
        print("✅ Good generalization! Test return >= 70% of train return")
        overfit_warning = False
    elif test_return >= train_return * 0.5:
        print("⚠️ Moderate overfitting. Test return is 50-70% of train return")
        overfit_warning = True
    else:
        print("❌ Significant overfitting! Test return < 50% of train return")
        overfit_warning = True
    
    return {
        'best_params': best_params.to_dict(),
        'train_results': train_results,
        'test_metrics': test_metrics,
        'overfit_warning': overfit_warning
    }


# =============================================================================
# TOP RESULTS DISPLAY
# =============================================================================

def show_top_results(results_df, metric='risk_adj_return', top_n=10):
    """Display top parameter combinations."""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} PARAMETER COMBINATIONS (by {metric})")
    print(f"{'='*80}")
    
    sorted_df = results_df.sort_values(metric, ascending=False).head(top_n)
    
    display_cols = ['smci_entry_min', 'smci_exit_threshold', 'weight_fg', 
                    'weight_pm', 'weight_fr', 'total_return', 'max_drawdown', 
                    'risk_adj_return', 'num_trades']
    
    print(sorted_df[display_cols].to_string(index=False))
    
    return sorted_df


# =============================================================================
# ROBUSTNESS CHECK
# =============================================================================

def check_robustness(results_df, best_params):
    """
    Check if nearby parameters also perform well.
    Good parameters should be "robust" - nearby values also work.
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS CHECK")
    print("=" * 60)
    
    entry_min = best_params['smci_entry_min']
    
    # Find results with similar entry_min (±5)
    nearby = results_df[
        (results_df['smci_entry_min'] >= entry_min - 5) &
        (results_df['smci_entry_min'] <= entry_min + 5)
    ]
    
    if len(nearby) > 1:
        avg_return = nearby['total_return'].mean()
        std_return = nearby['total_return'].std()
        
        print(f"Nearby parameters (entry_min ±5):")
        print(f"  Count:       {len(nearby)}")
        print(f"  Avg Return:  {avg_return:.2f}%")
        print(f"  Std Return:  {std_return:.2f}%")
        
        if std_return < avg_return * 0.3:
            print("✅ Parameters are robust (low variance)")
        else:
            print("⚠️ Parameters may be unstable (high variance)")
    else:
        print("Not enough nearby parameters to check robustness")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main optimization function."""
    print("=" * 60)
    print("STRATEGY C PARAMETER OPTIMIZER")
    print("With Walk-Forward Validation")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data('SOL_1h_data.csv')
    
    # Run walk-forward optimization
    results = walk_forward_optimize(df, PARAM_GRID, train_ratio=TRAIN_RATIO)
    
    # Show top results from training
    show_top_results(results['train_results'], metric='risk_adj_return', top_n=10)
    show_top_results(results['train_results'], metric='total_return', top_n=10)
    
    # Check robustness
    check_robustness(results['train_results'], results['best_params'])
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    
    bp = results['best_params']
    tm = results['test_metrics']
    
    print(f"""
// Optimized Strategy C Parameters
// Walk-Forward Validated (Train: 70%, Test: 30%)

const SMCI_ENTRY_MIN = {bp['smci_entry_min']};
const SMCI_EXIT_THRESHOLD = {bp['smci_exit_threshold']};

const WEIGHT_FEAR_GREED = {bp['weight_fg']};
const WEIGHT_POLYMARKET = {bp['weight_pm']};
const WEIGHT_FUNDING = {bp['weight_fr']:.1f};

// Expected Performance:
// Training Return: {bp['total_return']:.2f}%
// Test Return: {tm['total_return']:.2f}%
// Test Max Drawdown: {tm['max_drawdown']:.2f}%
// Test Risk-Adj Return: {tm['risk_adj_return']:.2f}
""")
    
    if results['overfit_warning']:
        print("⚠️ WARNING: Possible overfitting detected.")
        print("   Consider using more conservative parameters.")
    else:
        print("✅ Parameters appear well-generalized.")
    
    print("\nDone!")
    
    return results


if __name__ == "__main__":
    results = main()
