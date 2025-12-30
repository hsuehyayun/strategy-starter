"""
Strategy C Parameter Optimizer (Simplified SMCI)
================================================
Removes Funding Rate from SMCI to reduce noise and overfitting.

Simplified SMCI Formula:
SMCI = weight_fg × Fear_Greed_Signal + weight_pm × Polymarket_Proxy

Rationale for removing Funding Rate:
- Previous optimization showed Funding Rate may introduce noise
- Best performing combinations had low Funding Rate weight (0.2)
- Simpler model = less prone to overfitting
"""

import pandas as pd
import numpy as np
from datetime import datetime
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

# Simplified parameter grid (NO Funding Rate)
PARAM_GRID = {
    #'smci_entry_min': [40, 45, 50, 55],
    'smci_entry_min': [35, 40, 45, 50, 55],
    #'smci_exit_threshold': [30, 35, 40],
    'smci_exit_threshold': [25, 30, 35, 40],
    'weight_fg': [0.4, 0.5, 0.6, 0.7],
    # weight_pm = 1.0 - weight_fg (automatically calculated)
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
        print(f"  F&G data loaded successfully")
    except Exception as e:
        print(f"  Could not fetch F&G: {e}, using placeholder")
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
    
    # Drop NaN rows
    df = df.dropna()
    
    print(f"  Prepared {len(df)} rows")
    return df


# =============================================================================
# SIMPLIFIED SMCI CALCULATION (NO Funding Rate)
# =============================================================================

def calculate_smci_simple(df, weight_fg):
    """
    Calculate simplified SMCI with only F&G and Polymarket Proxy.
    
    SMCI = weight_fg × F&G_Signal + (1-weight_fg) × Polymarket_Proxy
    """
    weight_pm = 1.0 - weight_fg
    
    smci = (
        weight_fg * df['fg_signal'] +
        weight_pm * df['polymarket_proxy']
    )
    return smci.clip(0, 100).ewm(span=24).mean()


# =============================================================================
# BACKTESTER
# =============================================================================

def run_backtest(df, smci_entry_min, smci_exit_threshold, weight_fg):
    """
    Run backtest with given parameters.
    """
    df = df.copy()
    
    # Calculate simplified SMCI
    df['smci'] = calculate_smci_simple(df, weight_fg)
    
    # Generate signals
    sma_bullish = (df['sma_short'] > df['sma_long']).astype(int)
    sma_cross_up = sma_bullish.diff() == 1
    sma_cross_down = sma_bullish.diff() == -1
    
    rsi_ok = (df['rsi'] > RSI_OVERSOLD) & (df['rsi'] < RSI_OVERBOUGHT)
    rsi_overbought = df['rsi'] > 75
    
    smci_bullish = df['smci'] > smci_entry_min
    smci_bearish = df['smci'] < smci_exit_threshold
    
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
    
    if trades:
        wins = [t for t in trades if t > 0]
        win_rate = len(wins) / len(trades) * 100
    else:
        win_rate = 0
    
    equity_series = pd.Series(equity_history)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()
    
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
    """Grid search for best parameters."""
    print("\nOptimizing parameters on training data...")
    
    results = []
    
    entry_mins = param_grid['smci_entry_min']
    exit_thresholds = param_grid['smci_exit_threshold']
    weight_fgs = param_grid['weight_fg']
    
    total = len(entry_mins) * len(exit_thresholds) * len(weight_fgs)
    print(f"  Testing up to {total} parameter combinations...")
    
    count = 0
    for entry_min in entry_mins:
        for exit_thresh in exit_thresholds:
            for w_fg in weight_fgs:
                
                # Skip invalid combinations
                if exit_thresh >= entry_min:
                    continue
                
                metrics = run_backtest(
                    df_train,
                    smci_entry_min=entry_min,
                    smci_exit_threshold=exit_thresh,
                    weight_fg=w_fg
                )
                
                results.append({
                    'smci_entry_min': entry_min,
                    'smci_exit_threshold': exit_thresh,
                    'weight_fg': w_fg,
                    'weight_pm': 1.0 - w_fg,
                    **metrics
                })
                
                count += 1
                if count % 20 == 0:
                    print(f"  Completed {count} tests...")
    
    results_df = pd.DataFrame(results)
    print(f"  Tested {len(results_df)} valid combinations")
    
    return results_df


def walk_forward_optimize(df, param_grid, train_ratio=0.7):
    """Walk-forward optimization with train/test split."""
    
    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    print(f"\n{'='*60}")
    print("DATA SPLIT")
    print(f"{'='*60}")
    print(f"Train: {df_train.index.min().date()} to {df_train.index.max().date()} ({len(df_train)} rows)")
    print(f"Test:  {df_test.index.min().date()} to {df_test.index.max().date()} ({len(df_test)} rows)")
    
    # Optimize on training data
    train_results = optimize_parameters(df_train, param_grid)
    
    # Find best by risk-adjusted return
    best_idx = train_results['risk_adj_return'].idxmax()
    best_params = train_results.loc[best_idx]
    
    # Also find best by total return
    best_return_idx = train_results['total_return'].idxmax()
    best_return_params = train_results.loc[best_return_idx]
    
    print("\n" + "=" * 60)
    print("BEST BY RISK-ADJUSTED RETURN (Training)")
    print("=" * 60)
    print(f"  SMCI Entry Min:     {best_params['smci_entry_min']}")
    print(f"  SMCI Exit Threshold: {best_params['smci_exit_threshold']}")
    print(f"  Weight F&G:         {best_params['weight_fg']}")
    print(f"  Weight Polymarket:  {best_params['weight_pm']}")
    print(f"\n  Training: Return={best_params['total_return']:.2f}%, MaxDD={best_params['max_drawdown']:.2f}%")
    
    # Test best risk-adj params
    test_metrics_risk = run_backtest(
        df_test,
        smci_entry_min=best_params['smci_entry_min'],
        smci_exit_threshold=best_params['smci_exit_threshold'],
        weight_fg=best_params['weight_fg']
    )
    print(f"  Test:     Return={test_metrics_risk['total_return']:.2f}%, MaxDD={test_metrics_risk['max_drawdown']:.2f}%")
    
    print("\n" + "=" * 60)
    print("BEST BY TOTAL RETURN (Training)")
    print("=" * 60)
    print(f"  SMCI Entry Min:     {best_return_params['smci_entry_min']}")
    print(f"  SMCI Exit Threshold: {best_return_params['smci_exit_threshold']}")
    print(f"  Weight F&G:         {best_return_params['weight_fg']}")
    print(f"  Weight Polymarket:  {best_return_params['weight_pm']}")
    print(f"\n  Training: Return={best_return_params['total_return']:.2f}%, MaxDD={best_return_params['max_drawdown']:.2f}%")
    
    # Test best return params
    test_metrics_return = run_backtest(
        df_test,
        smci_entry_min=best_return_params['smci_entry_min'],
        smci_exit_threshold=best_return_params['smci_exit_threshold'],
        weight_fg=best_return_params['weight_fg']
    )
    print(f"  Test:     Return={test_metrics_return['total_return']:.2f}%, MaxDD={test_metrics_return['max_drawdown']:.2f}%")
    
    # Overfitting check for both
    print("\n" + "=" * 60)
    print("OVERFITTING CHECK")
    print("=" * 60)
    
    ratio_risk = test_metrics_risk['total_return'] / best_params['total_return'] if best_params['total_return'] > 0 else 0
    ratio_return = test_metrics_return['total_return'] / best_return_params['total_return'] if best_return_params['total_return'] > 0 else 0
    
    print(f"\nBest Risk-Adj: Test/Train ratio = {ratio_risk:.2%}")
    if ratio_risk >= 0.7:
        print("  ✅ Good generalization!")
    elif ratio_risk >= 0.5:
        print("  ⚠️ Moderate overfitting")
    else:
        print("  ❌ Significant overfitting")
    
    print(f"\nBest Return: Test/Train ratio = {ratio_return:.2%}")
    if ratio_return >= 0.7:
        print("  ✅ Good generalization!")
    elif ratio_return >= 0.5:
        print("  ⚠️ Moderate overfitting")
    else:
        print("  ❌ Significant overfitting")
    
    return {
        'train_results': train_results,
        'best_risk_adj': {
            'params': best_params.to_dict(),
            'test_metrics': test_metrics_risk
        },
        'best_return': {
            'params': best_return_params.to_dict(),
            'test_metrics': test_metrics_return
        }
    }


# =============================================================================
# COMPARISON WITH STRATEGY A & B
# =============================================================================

def run_strategy_a(df):
    """Run Strategy A for comparison."""
    sma_bullish = (df['sma_short'] > df['sma_long']).astype(int)
    sma_cross_up = sma_bullish.diff() == 1
    sma_cross_down = sma_bullish.diff() == -1
    
    rsi_ok = (df['rsi'] > RSI_OVERSOLD) & (df['rsi'] < RSI_OVERBOUGHT)
    rsi_overbought = df['rsi'] > 75
    
    buy_signal = sma_cross_up & rsi_ok
    sell_signal = sma_cross_down | rsi_overbought
    
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    equity_history = [INITIAL_CAPITAL]
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        
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
    
    final_equity = capital + position * df['close'].iloc[-1]
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    equity_series = pd.Series(equity_history)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    return {'total_return': total_return, 'max_drawdown': max_drawdown, 'num_trades': len(trades)}


def run_strategy_b(df):
    """Run Strategy B for comparison."""
    sma_bullish = (df['sma_short'] > df['sma_long']).astype(int)
    sma_cross_up = sma_bullish.diff() == 1
    sma_cross_down = sma_bullish.diff() == -1
    
    rsi_ok = (df['rsi'] > RSI_OVERSOLD) & (df['rsi'] < RSI_OVERBOUGHT)
    rsi_overbought = df['rsi'] > 75
    
    fg_ok = df['fear_greed'] < 75
    fg_extreme = df['fear_greed'] > 80
    
    buy_signal = sma_cross_up & rsi_ok & fg_ok
    sell_signal = sma_cross_down | rsi_overbought | fg_extreme
    
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    equity_history = [INITIAL_CAPITAL]
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        
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
    
    final_equity = capital + position * df['close'].iloc[-1]
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    equity_series = pd.Series(equity_history)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    return {'total_return': total_return, 'max_drawdown': max_drawdown, 'num_trades': len(trades)}


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

def show_top_results(results_df, metric='risk_adj_return', top_n=10):
    """Display top parameter combinations."""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} BY {metric.upper()}")
    print(f"{'='*80}")
    
    sorted_df = results_df.sort_values(metric, ascending=False).head(top_n)
    
    display_cols = ['smci_entry_min', 'smci_exit_threshold', 'weight_fg', 
                    'weight_pm', 'total_return', 'max_drawdown', 
                    'risk_adj_return', 'num_trades']
    
    print(sorted_df[display_cols].to_string(index=False))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main optimization function."""
    print("=" * 60)
    print("SIMPLIFIED SMCI OPTIMIZER")
    print("(Without Funding Rate)")
    print("=" * 60)
    print("\nFormula: SMCI = weight_fg × F&G + weight_pm × Polymarket")
    
    # Load data
    df = load_and_prepare_data('SOL_1h_data.csv')
    
    # Run walk-forward optimization
    results = walk_forward_optimize(df, PARAM_GRID, train_ratio=TRAIN_RATIO)
    
    # Show top results
    show_top_results(results['train_results'], 'risk_adj_return', 10)
    show_top_results(results['train_results'], 'total_return', 10)
    
    # Compare with Strategy A & B on TEST data
    print("\n" + "=" * 60)
    print("COMPARISON WITH STRATEGY A & B (Full Data)")
    print("=" * 60)
    
    results_a = run_strategy_a(df)
    results_b = run_strategy_b(df)
    
    best_c_params = results['best_return']['params']
    results_c = run_backtest(
        df,
        smci_entry_min=best_c_params['smci_entry_min'],
        smci_exit_threshold=best_c_params['smci_exit_threshold'],
        weight_fg=best_c_params['weight_fg']
    )
    
    print(f"\n{'Strategy':<15} {'Return':>12} {'Max DD':>12} {'Risk-Adj':>12}")
    print("-" * 55)
    print(f"{'Strategy A':<15} {results_a['total_return']:>11.2f}% {results_a['max_drawdown']:>11.2f}% {results_a['total_return']/abs(results_a['max_drawdown']):>12.2f}")
    print(f"{'Strategy B':<15} {results_b['total_return']:>11.2f}% {results_b['max_drawdown']:>11.2f}% {results_b['total_return']/abs(results_b['max_drawdown']):>12.2f}")
    print(f"{'Strategy C (new)':<15} {results_c['total_return']:>11.2f}% {results_c['max_drawdown']:>11.2f}% {results_c['risk_adj_return']:>12.2f}")
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    
    best = results['best_return']
    bp = best['params']
    tm = best['test_metrics']
    
    print(f"""
// Simplified Strategy C Parameters
// SMCI = {bp['weight_fg']} × F&G + {bp['weight_pm']:.1f} × Polymarket
// (No Funding Rate)

const SMCI_ENTRY_MIN = {bp['smci_entry_min']};
const SMCI_EXIT_THRESHOLD = {bp['smci_exit_threshold']};

const WEIGHT_FEAR_GREED = {bp['weight_fg']};
const WEIGHT_POLYMARKET = {bp['weight_pm']:.1f};

// Training Performance: {bp['total_return']:.2f}%
// Test Performance:     {tm['total_return']:.2f}%
// Test Max Drawdown:    {tm['max_drawdown']:.2f}%
""")
    
    # Determine winner
    all_returns = {
        'A': results_a['total_return'],
        'B': results_b['total_return'],
        'C': results_c['total_return']
    }
    winner = max(all_returns, key=all_returns.get)
    print(f"🏆 WINNER (Full Data): Strategy {winner} with {all_returns[winner]:.2f}% return")
    
    print("\nDone!")
    
    return results


if __name__ == "__main__":
    results = main()
