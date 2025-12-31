"""
Exploratory Data Analysis (EDA) for SOL Trading Data
=====================================================
This script analyzes the SOL price data to:
1. Understand basic statistics
2. Identify bull/bear market periods
3. Analyze volatility patterns
4. Check correlation with Fear & Greed
5. Provide insights for strategy selection

Run this BEFORE optimizing parameters!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING
# =============================================================================

def load_sol_data(filepath='SOL_1h_data.csv'):
    """Load SOL price data."""
    print("Loading SOL data...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"  Loaded {len(df)} rows")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def fetch_fear_greed():
    """Fetch Fear & Greed Index."""
    print("Fetching Fear & Greed data...")
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, params={'limit': 0}, timeout=30)
        data = response.json()['data']
        
        records = []
        for item in data:
            records.append({
                'date': datetime.fromtimestamp(int(item['timestamp'])),
                'fear_greed': int(item['value'])
            })
        
        fg_df = pd.DataFrame(records).set_index('date').sort_index()
        print(f"  Loaded {len(fg_df)} days")
        return fg_df
    except Exception as e:
        print(f"  Error: {e}")
        return None


# =============================================================================
# BASIC STATISTICS
# =============================================================================

def basic_statistics(df):
    """Calculate basic price statistics."""
    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    
    # Price stats
    print(f"\nPrice Statistics:")
    print(f"  Start Price:  ${df['close'].iloc[0]:.2f}")
    print(f"  End Price:    ${df['close'].iloc[-1]:.2f}")
    print(f"  Min Price:    ${df['close'].min():.2f}")
    print(f"  Max Price:    ${df['close'].max():.2f}")
    print(f"  Mean Price:   ${df['close'].mean():.2f}")
    print(f"  Median Price: ${df['close'].median():.2f}")
    
    # Returns
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"\n  Total Return: {total_return:+.2f}%")
    
    # Daily returns
    daily_df = df['close'].resample('D').last().dropna()
    daily_returns = daily_df.pct_change().dropna()
    
    print(f"\nDaily Return Statistics:")
    print(f"  Mean:   {daily_returns.mean()*100:+.3f}%")
    print(f"  Std:    {daily_returns.std()*100:.3f}%")
    print(f"  Min:    {daily_returns.min()*100:.2f}%")
    print(f"  Max:    {daily_returns.max()*100:+.2f}%")
    print(f"  Sharpe: {daily_returns.mean()/daily_returns.std()*np.sqrt(365):.2f} (annualized)")
    
    # Volume
    if 'volume' in df.columns:
        print(f"\nVolume Statistics:")
        print(f"  Mean Daily Volume: ${df['volume'].resample('D').sum().mean():,.0f}")
    
    return daily_returns


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

def detect_market_regimes(df):
    """
    Detect bull/bear market periods using multiple methods.
    """
    print("\n" + "=" * 60)
    print("MARKET REGIME ANALYSIS")
    print("=" * 60)
    
    # Resample to daily
    daily = df['close'].resample('D').last().dropna()
    
    # Method 1: 50-day and 200-day MA
    ma_50 = daily.rolling(50).mean()
    ma_200 = daily.rolling(200).mean()
    
    # Bull: Price > MA200 and MA50 > MA200
    # Bear: Price < MA200 and MA50 < MA200
    regime = pd.Series(index=daily.index, dtype='object')
    regime[:] = 'NEUTRAL'
    
    bull_mask = (daily > ma_200) & (ma_50 > ma_200)
    bear_mask = (daily < ma_200) & (ma_50 < ma_200)
    
    regime[bull_mask] = 'BULL'
    regime[bear_mask] = 'BEAR'
    
    # Count days in each regime
    regime_counts = regime.value_counts()
    total_days = len(regime.dropna())
    
    print("\nMarket Regime Distribution:")
    for r in ['BULL', 'NEUTRAL', 'BEAR']:
        if r in regime_counts:
            count = regime_counts[r]
            pct = count / total_days * 100
            print(f"  {r}: {count} days ({pct:.1f}%)")
    
    # Identify regime periods
    print("\nMajor Market Periods:")
    
    # Find regime changes
    regime_changes = regime != regime.shift(1)
    change_dates = regime.index[regime_changes]
    
    periods = []
    for i in range(len(change_dates)):
        start = change_dates[i]
        end = change_dates[i+1] if i+1 < len(change_dates) else regime.index[-1]
        r = regime.loc[start]
        duration = (end - start).days
        
        if duration >= 30:  # Only show periods >= 30 days
            start_price = daily.loc[start]
            end_price = daily.loc[end] if end in daily.index else daily.iloc[-1]
            ret = (end_price / start_price - 1) * 100
            
            periods.append({
                'start': start,
                'end': end,
                'regime': r,
                'duration': duration,
                'return': ret
            })
            print(f"  {start.date()} to {end.date()}: {r} ({duration} days, {ret:+.1f}%)")
    
    return regime, periods


def analyze_regime_characteristics(df, regime):
    """
    Analyze how price behaves differently in bull vs bear markets.
    """
    print("\n" + "=" * 60)
    print("REGIME CHARACTERISTICS")
    print("=" * 60)
    
    # Merge regime with hourly data
    daily = df['close'].resample('D').last().dropna()
    df_daily = pd.DataFrame({'close': daily})
    df_daily['regime'] = regime
    df_daily['return'] = df_daily['close'].pct_change()
    
    for r in ['BULL', 'BEAR', 'NEUTRAL']:
        subset = df_daily[df_daily['regime'] == r]['return'].dropna()
        if len(subset) > 10:
            print(f"\n{r} Market:")
            print(f"  Days:       {len(subset)}")
            print(f"  Avg Return: {subset.mean()*100:+.3f}%/day")
            print(f"  Volatility: {subset.std()*100:.3f}%/day")
            print(f"  Win Rate:   {(subset > 0).mean()*100:.1f}%")
            print(f"  Best Day:   {subset.max()*100:+.2f}%")
            print(f"  Worst Day:  {subset.min()*100:.2f}%")


# =============================================================================
# VOLATILITY ANALYSIS
# =============================================================================

def analyze_volatility(df):
    """Analyze volatility patterns."""
    print("\n" + "=" * 60)
    print("VOLATILITY ANALYSIS")
    print("=" * 60)
    
    # Calculate rolling volatility (20-day)
    daily = df['close'].resample('D').last().dropna()
    returns = daily.pct_change().dropna()
    
    vol_20d = returns.rolling(20).std() * np.sqrt(365) * 100  # Annualized
    
    print(f"\nAnnualized Volatility (20-day rolling):")
    print(f"  Current: {vol_20d.iloc[-1]:.1f}%")
    print(f"  Mean:    {vol_20d.mean():.1f}%")
    print(f"  Min:     {vol_20d.min():.1f}%")
    print(f"  Max:     {vol_20d.max():.1f}%")
    
    # Volatility by year
    print(f"\nVolatility by Year:")
    for year in returns.index.year.unique():
        year_returns = returns[returns.index.year == year]
        year_vol = year_returns.std() * np.sqrt(365) * 100
        print(f"  {year}: {year_vol:.1f}%")
    
    # Volatility by month
    monthly_vol = returns.groupby(returns.index.month).std() * np.sqrt(365) * 100
    print(f"\nVolatility by Month (average):")
    high_vol_months = monthly_vol.nlargest(3)
    low_vol_months = monthly_vol.nsmallest(3)
    print(f"  Highest: {', '.join([f'Month {m} ({v:.0f}%)' for m, v in high_vol_months.items()])}")
    print(f"  Lowest:  {', '.join([f'Month {m} ({v:.0f}%)' for m, v in low_vol_months.items()])}")
    
    return vol_20d


# =============================================================================
# FEAR & GREED CORRELATION
# =============================================================================

def analyze_fear_greed_correlation(df, fg_df):
    """Analyze correlation between price and Fear & Greed."""
    print("\n" + "=" * 60)
    print("FEAR & GREED ANALYSIS")
    print("=" * 60)
    
    if fg_df is None:
        print("  No Fear & Greed data available")
        return
    
    # Merge data
    daily = df['close'].resample('D').last().dropna()
    daily_returns = daily.pct_change()
    
    merged = pd.DataFrame({
        'price': daily,
        'return': daily_returns,
        'fg': fg_df['fear_greed']
    }).dropna()
    
    print(f"\nFear & Greed Statistics:")
    print(f"  Mean:    {merged['fg'].mean():.1f}")
    print(f"  Current: {merged['fg'].iloc[-1]:.0f}")
    
    # Correlation
    corr_price = merged['fg'].corr(merged['price'])
    corr_return = merged['fg'].corr(merged['return'])
    
    print(f"\nCorrelations:")
    print(f"  F&G vs Price:  {corr_price:.3f}")
    print(f"  F&G vs Return: {corr_return:.3f}")
    
    # Performance by F&G level
    print(f"\nNext-Day Returns by F&G Level:")
    
    merged['fg_level'] = pd.cut(merged['fg'], 
                                bins=[0, 25, 45, 55, 75, 100],
                                labels=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'])
    merged['next_return'] = merged['return'].shift(-1)
    
    fg_performance = merged.groupby('fg_level')['next_return'].agg(['mean', 'std', 'count'])
    fg_performance['mean'] = fg_performance['mean'] * 100
    fg_performance['std'] = fg_performance['std'] * 100
    
    for level in ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']:
        if level in fg_performance.index:
            row = fg_performance.loc[level]
            print(f"  {level:14}: {row['mean']:+.3f}%/day (n={row['count']:.0f})")
    
    # Trading signal analysis
    print(f"\nContrarian Signal Analysis:")
    
    # Buy on Extreme Fear
    extreme_fear = merged[merged['fg'] < 25]['next_return'].dropna()
    if len(extreme_fear) > 0:
        print(f"  Buy on Extreme Fear (<25):")
        print(f"    Avg Next-Day Return: {extreme_fear.mean()*100:+.3f}%")
        print(f"    Win Rate: {(extreme_fear > 0).mean()*100:.1f}%")
    
    # Sell on Extreme Greed
    extreme_greed = merged[merged['fg'] > 75]['next_return'].dropna()
    if len(extreme_greed) > 0:
        print(f"  Sell on Extreme Greed (>75):")
        print(f"    Avg Next-Day Return: {extreme_greed.mean()*100:+.3f}%")
        print(f"    Win Rate: {(extreme_greed > 0).mean()*100:.1f}%")


# =============================================================================
# RECENT PERIOD ANALYSIS (Competition Period Proxy)
# =============================================================================

def analyze_recent_period(df, days=60):
    """Analyze recent period as proxy for competition."""
    print("\n" + "=" * 60)
    print(f"RECENT {days} DAYS ANALYSIS (Competition Proxy)")
    print("=" * 60)
    
    recent = df.iloc[-days*24:]  # Last N days of hourly data
    
    print(f"\nDate Range: {recent.index.min().date()} to {recent.index.max().date()}")
    
    # Price movement
    start_price = recent['close'].iloc[0]
    end_price = recent['close'].iloc[-1]
    ret = (end_price / start_price - 1) * 100
    
    print(f"\nPrice Movement:")
    print(f"  Start: ${start_price:.2f}")
    print(f"  End:   ${end_price:.2f}")
    print(f"  Return: {ret:+.2f}%")
    
    # Volatility
    daily_recent = recent['close'].resample('D').last().dropna()
    daily_returns = daily_recent.pct_change().dropna()
    vol = daily_returns.std() * np.sqrt(365) * 100
    
    print(f"\nVolatility: {vol:.1f}% (annualized)")
    
    # Trend strength
    ma_10 = recent['close'].rolling(10*24).mean()
    ma_50 = recent['close'].rolling(50*24).mean()
    
    trend_strength = (ma_10.iloc[-1] / ma_50.iloc[-1] - 1) * 100
    print(f"Trend Strength (MA10/MA50): {trend_strength:+.2f}%")
    
    if trend_strength > 5:
        print("  → Strong UPTREND")
    elif trend_strength > 0:
        print("  → Mild uptrend")
    elif trend_strength > -5:
        print("  → Mild downtrend")
    else:
        print("  → Strong DOWNTREND")
    
    return recent


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_eda(df, regime, vol_20d, fg_df):
    """Create comprehensive EDA visualization."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 18),
                             gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})
    
    daily = df['close'].resample('D').last().dropna()
    
    # 1. Price with Market Regimes
    ax1 = axes[0]
    ax1.plot(daily.index, daily.values, color='#2c3e50', linewidth=1)
    
    # Color background by regime
    for i in range(len(regime)-1):
        if regime.iloc[i] == 'BULL':
            ax1.axvspan(regime.index[i], regime.index[i+1], alpha=0.2, color='green')
        elif regime.iloc[i] == 'BEAR':
            ax1.axvspan(regime.index[i], regime.index[i+1], alpha=0.2, color='red')
    
    # Add MAs
    ma_50 = daily.rolling(50).mean()
    ma_200 = daily.rolling(200).mean()
    ax1.plot(ma_50.index, ma_50.values, color='orange', linewidth=1, label='MA50', alpha=0.7)
    ax1.plot(ma_200.index, ma_200.values, color='purple', linewidth=1, label='MA200', alpha=0.7)
    
    ax1.set_title('SOL Price with Market Regimes (Green=Bull, Red=Bear)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Daily Returns Distribution
    ax2 = axes[1]
    returns = daily.pct_change().dropna() * 100
    ax2.hist(returns, bins=100, color='#3498db', alpha=0.7, edgecolor='white')
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.axvline(x=returns.mean(), color='green', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
    ax2.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Daily Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling Volatility
    ax3 = axes[2]
    ax3.fill_between(vol_20d.index, 0, vol_20d.values, color='#e74c3c', alpha=0.5)
    ax3.axhline(y=vol_20d.mean(), color='blue', linestyle='--', label=f'Mean: {vol_20d.mean():.0f}%')
    ax3.set_title('Annualized Volatility (20-day Rolling)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Volatility (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Fear & Greed Index
    ax4 = axes[3]
    if fg_df is not None:
        ax4.fill_between(fg_df.index, 0, fg_df['fear_greed'], color='#9b59b6', alpha=0.5)
        ax4.axhline(y=25, color='green', linestyle='--', alpha=0.7, label='Extreme Fear (25)')
        ax4.axhline(y=75, color='red', linestyle='--', alpha=0.7, label='Extreme Greed (75)')
        ax4.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        ax4.set_ylim(0, 100)
        ax4.legend(loc='upper left')
    ax4.set_title('Fear & Greed Index', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F&G Score')
    ax4.grid(True, alpha=0.3)
    
    # 5. Monthly Returns Heatmap-style bar chart
    ax5 = axes[4]
    monthly_returns = daily.resample('M').last().pct_change() * 100
    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in monthly_returns.values]
    ax5.bar(monthly_returns.index, monthly_returns.values, width=20, color=colors, alpha=0.7)
    ax5.axhline(y=0, color='black', linewidth=0.5)
    ax5.set_title('Monthly Returns', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Return (%)')
    ax5.set_xlabel('Date')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nChart saved: eda_analysis.png")


# =============================================================================
# STRATEGY RECOMMENDATION
# =============================================================================

def strategy_recommendation(df, regime, fg_df):
    """Provide strategy recommendation based on analysis."""
    print("\n" + "=" * 60)
    print("STRATEGY RECOMMENDATION")
    print("=" * 60)
    
    # Current market state
    daily = df['close'].resample('D').last().dropna()
    current_regime = regime.iloc[-1] if len(regime) > 0 else 'UNKNOWN'
    
    ma_50 = daily.rolling(50).mean().iloc[-1]
    ma_200 = daily.rolling(200).mean().iloc[-1]
    current_price = daily.iloc[-1]
    
    print(f"\nCurrent Market State:")
    print(f"  Price:  ${current_price:.2f}")
    print(f"  MA50:   ${ma_50:.2f}")
    print(f"  MA200:  ${ma_200:.2f}")
    print(f"  Regime: {current_regime}")
    
    if fg_df is not None:
        current_fg = fg_df['fear_greed'].iloc[-1]
        print(f"  F&G:    {current_fg}")
    else:
        current_fg = 50
    
    # Recommendation
    print(f"\n📊 RECOMMENDATION FOR COMPETITION:")
    
    if current_regime == 'BULL':
        print("  Market is in BULL regime")
        print("  → Consider Strategy C (more aggressive)")
        print("  → Higher SMCI entry threshold OK")
    elif current_regime == 'BEAR':
        print("  Market is in BEAR regime")
        print("  → Consider Strategy B (more conservative)")
        print("  → Focus on risk management")
    else:
        print("  Market is in NEUTRAL/TRANSITION regime")
        print("  → Consider Strategy B (balanced)")
        print("  → Be prepared for volatility")
    
    if current_fg < 30:
        print(f"\n  ⚠️ F&G is low ({current_fg}) - potential buying opportunity")
    elif current_fg > 70:
        print(f"\n  ⚠️ F&G is high ({current_fg}) - be cautious of reversal")
    
    # Competition period consideration
    print(f"\n📅 COMPETITION TIMING (Jan 1-14):")
    print("  - Holiday period: Lower liquidity")
    print("  - Tax-loss selling may be ending")
    print("  - Potential 'January effect' rally")
    print("  - Watch for New Year volatility")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run full EDA analysis."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("SOL Trading Data")
    print("=" * 60)
    
    # Load data
    df = load_sol_data('SOL_1h_data.csv')
    fg_df = fetch_fear_greed()
    
    # Basic statistics
    daily_returns = basic_statistics(df)
    
    # Market regime detection
    regime, periods = detect_market_regimes(df)
    
    # Regime characteristics
    analyze_regime_characteristics(df, regime)
    
    # Volatility analysis
    vol_20d = analyze_volatility(df)
    
    # Fear & Greed correlation
    analyze_fear_greed_correlation(df, fg_df)
    
    # Recent period analysis
    analyze_recent_period(df, days=60)
    
    # Strategy recommendation
    strategy_recommendation(df, regime, fg_df)
    
    # Visualization
    print("\nGenerating charts...")
    try:
        plot_eda(df, regime, vol_20d, fg_df)
    except Exception as e:
        print(f"  Could not generate chart: {e}")
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the analysis above")
    print("2. Check the eda_analysis.png chart")
    print("3. Decide on strategy based on current market regime")
    print("4. Run parameter optimization with appropriate settings")
    
    return df, regime, fg_df


if __name__ == "__main__":
    df, regime, fg_df = main()
