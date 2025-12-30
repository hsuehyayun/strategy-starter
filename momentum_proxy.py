"""
Momentum Proxy for Polymarket Sentiment
=======================================
This module calculates price momentum as a proxy for Polymarket prediction
market sentiment, since historical Polymarket data is not readily available.

Theoretical Foundation:
-----------------------
1. Extrapolation Bias (Barberis, Shleifer & Vishny, 1998):
   - Investors tend to extrapolate recent price trends into the future
   - When prices rise, people expect continued rises
   - This is reflected in both prediction market prices AND price momentum

2. Greenwood & Shleifer (2014) - "Expectations of Returns and Expected Returns":
   - Survey-based expectations are highly correlated with past returns
   - Correlation > 0.7 between recent returns and future expectations
   - This validates using momentum as a proxy for market expectations

3. Wisdom of Crowds (Surowiecki, 2004):
   - Both prediction markets and price momentum aggregate dispersed information
   - Price is a "voting mechanism" where money = votes

Proxy Design:
-------------
We use a combination of momentum indicators to estimate what Polymarket
sentiment would likely show:

1. Rate of Change (ROC): Direct measure of price momentum
2. Momentum RSI: Strength of the trend, normalized 0-100
3. Moving Average Trend: Direction and strength of trend

The final "Polymarket Proxy Score" is weighted average of these indicators,
scaled to 0-100 to match probability-style interpretation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default lookback periods
ROC_PERIOD = 14          # 14 days for rate of change
MOMENTUM_RSI_PERIOD = 14 # 14 days for momentum RSI
MA_SHORT = 10            # Short-term MA
MA_LONG = 30             # Long-term MA

# Weights for combining indicators (can be optimized)
WEIGHT_ROC = 0.4         # Rate of change weight
WEIGHT_MRSI = 0.4        # Momentum RSI weight  
WEIGHT_MA_TREND = 0.2    # MA trend weight

# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================

def calculate_roc(prices, period=14):
    """
    Calculate Rate of Change (ROC).
    
    ROC = ((Price_today - Price_n_days_ago) / Price_n_days_ago) * 100
    
    Interpretation:
    - Positive ROC: Price is higher than n days ago (uptrend)
    - Negative ROC: Price is lower than n days ago (downtrend)
    - Magnitude indicates strength of the move
    
    Args:
        prices: Series of prices
        period: Lookback period in days/candles
        
    Returns:
        Series of ROC values (percentage)
    """
    roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
    return roc


def calculate_momentum_rsi(prices, period=14):
    """
    Calculate Momentum RSI - a modified RSI that measures trend strength.
    
    This is similar to standard RSI but interpreted differently:
    - Values near 100: Strong upward momentum
    - Values near 50: No clear momentum
    - Values near 0: Strong downward momentum
    
    Args:
        prices: Series of prices
        period: RSI calculation period
        
    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_ma_trend_score(prices, short_period=10, long_period=30):
    """
    Calculate a trend score based on moving average relationship.
    
    Logic:
    - MA_short > MA_long: Uptrend → Higher score
    - MA_short < MA_long: Downtrend → Lower score
    - Distance between MAs indicates trend strength
    
    The score is normalized to 0-100.
    
    Args:
        prices: Series of prices
        short_period: Short MA period
        long_period: Long MA period
        
    Returns:
        Series of trend scores (0-100)
    """
    ma_short = prices.rolling(window=short_period).mean()
    ma_long = prices.rolling(window=long_period).mean()
    
    # Calculate percentage difference
    ma_diff_pct = ((ma_short - ma_long) / ma_long) * 100
    
    # Normalize to 0-100 scale
    # Assume ±10% difference is the typical range
    normalized = 50 + (ma_diff_pct / 10) * 50
    
    # Clip to 0-100
    normalized = normalized.clip(0, 100)
    
    return normalized


def calculate_price_momentum_score(prices, period=7):
    """
    Calculate a simple momentum score based on recent price change.
    
    This directly measures "how much has price moved recently"
    and converts it to a 0-100 score.
    
    Args:
        prices: Series of prices
        period: Lookback period
        
    Returns:
        Series of momentum scores (0-100)
    """
    returns = prices.pct_change(period) * 100
    
    # Normalize: assume ±20% weekly move is extreme
    normalized = 50 + (returns / 20) * 50
    
    return normalized.clip(0, 100)


# =============================================================================
# POLYMARKET PROXY CALCULATION
# =============================================================================

def calculate_polymarket_proxy(prices, 
                                roc_period=ROC_PERIOD,
                                mrsi_period=MOMENTUM_RSI_PERIOD,
                                ma_short=MA_SHORT,
                                ma_long=MA_LONG,
                                weight_roc=WEIGHT_ROC,
                                weight_mrsi=WEIGHT_MRSI,
                                weight_ma=WEIGHT_MA_TREND):
    """
    Calculate the Polymarket Sentiment Proxy score.
    
    This combines multiple momentum indicators to estimate what 
    prediction market sentiment would likely show.
    
    The output is a score from 0-100:
    - 0-30: Strong bearish sentiment (market expects decline)
    - 30-45: Moderately bearish
    - 45-55: Neutral
    - 55-70: Moderately bullish
    - 70-100: Strong bullish sentiment (market expects rise)
    
    Args:
        prices: Series of prices (should be daily or hourly)
        roc_period: Period for Rate of Change
        mrsi_period: Period for Momentum RSI
        ma_short: Short MA period
        ma_long: Long MA period
        weight_roc: Weight for ROC in final score
        weight_mrsi: Weight for Momentum RSI
        weight_ma: Weight for MA trend
        
    Returns:
        DataFrame with individual indicators and final proxy score
    """
    df = pd.DataFrame(index=prices.index)
    df['price'] = prices
    
    # Calculate individual indicators
    
    # 1. Rate of Change (normalized to 0-100)
    roc = calculate_roc(prices, roc_period)
    # Normalize: assume ±30% ROC over period is extreme
    df['roc_raw'] = roc
    df['roc_score'] = (50 + (roc / 30) * 50).clip(0, 100)
    
    # 2. Momentum RSI (already 0-100)
    df['momentum_rsi'] = calculate_momentum_rsi(prices, mrsi_period)
    
    # 3. MA Trend Score (already 0-100)
    df['ma_trend_score'] = calculate_ma_trend_score(prices, ma_short, ma_long)
    
    # 4. Calculate weighted average (Polymarket Proxy)
    df['polymarket_proxy'] = (
        weight_roc * df['roc_score'] +
        weight_mrsi * df['momentum_rsi'] +
        weight_ma * df['ma_trend_score']
    )
    
    # 5. Apply smoothing to reduce noise (7-day EMA)
    df['polymarket_proxy_smoothed'] = df['polymarket_proxy'].ewm(span=7).mean()
    
    return df


def calculate_polymarket_signal(proxy_score):
    """
    Convert Polymarket proxy score to a trading signal interpretation.
    
    Args:
        proxy_score: Polymarket proxy value (0-100)
        
    Returns:
        Dictionary with signal details
    """
    if proxy_score >= 70:
        return {
            'signal': 'STRONG_BULLISH',
            'description': 'Market strongly expects price increase',
            'action': 'Consider adding to long positions',
            'score': proxy_score
        }
    elif proxy_score >= 55:
        return {
            'signal': 'BULLISH',
            'description': 'Market moderately expects price increase',
            'action': 'Favorable for long entries',
            'score': proxy_score
        }
    elif proxy_score >= 45:
        return {
            'signal': 'NEUTRAL',
            'description': 'Market has no clear directional expectation',
            'action': 'Wait for clearer signal',
            'score': proxy_score
        }
    elif proxy_score >= 30:
        return {
            'signal': 'BEARISH',
            'description': 'Market moderately expects price decrease',
            'action': 'Caution for long positions',
            'score': proxy_score
        }
    else:
        return {
            'signal': 'STRONG_BEARISH',
            'description': 'Market strongly expects price decrease',
            'action': 'Consider reducing exposure',
            'score': proxy_score
        }


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_proxy_correlation(prices, actual_sentiment=None):
    """
    Validate the proxy by checking correlation with future returns.
    
    A good sentiment proxy should have some predictive power:
    - High proxy score → followed by positive returns
    - Low proxy score → followed by negative returns
    
    Args:
        prices: Price series
        actual_sentiment: Optional actual sentiment data for comparison
        
    Returns:
        Dictionary with validation metrics
    """
    df = calculate_polymarket_proxy(prices)
    
    # Calculate forward returns (1-day, 7-day, 14-day)
    df['fwd_return_1d'] = prices.pct_change(1).shift(-1)
    df['fwd_return_7d'] = prices.pct_change(7).shift(-7)
    df['fwd_return_14d'] = prices.pct_change(14).shift(-14)
    
    # Calculate correlations
    correlations = {
        'proxy_vs_1d_return': df['polymarket_proxy'].corr(df['fwd_return_1d']),
        'proxy_vs_7d_return': df['polymarket_proxy'].corr(df['fwd_return_7d']),
        'proxy_vs_14d_return': df['polymarket_proxy'].corr(df['fwd_return_14d']),
    }
    
    # Calculate hit rate (high proxy → positive return)
    df['proxy_bullish'] = df['polymarket_proxy'] > 55
    df['return_positive_7d'] = df['fwd_return_7d'] > 0
    
    bullish_predictions = df[df['proxy_bullish'] == True]
    if len(bullish_predictions) > 0:
        hit_rate = (bullish_predictions['return_positive_7d'] == True).sum() / len(bullish_predictions)
        correlations['bullish_hit_rate_7d'] = hit_rate
    
    return correlations


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def resample_to_daily(hourly_prices):
    """
    Resample hourly prices to daily for momentum calculation.
    
    Using daily data reduces noise in momentum indicators.
    
    Args:
        hourly_prices: Series with hourly price data
        
    Returns:
        Series with daily closing prices
    """
    return hourly_prices.resample('D').last().dropna()


def align_proxy_to_hourly(daily_proxy, hourly_index):
    """
    Align daily proxy scores to hourly data.
    
    Each hour of a day gets the same proxy score.
    
    Args:
        daily_proxy: Series with daily proxy scores
        hourly_index: DatetimeIndex of hourly data
        
    Returns:
        Series with proxy scores aligned to hourly index
    """
    # Create a date column for mapping
    hourly_dates = hourly_index.date
    
    # Map daily values to hourly
    daily_proxy_dated = daily_proxy.copy()
    daily_proxy_dated.index = daily_proxy_dated.index.date
    
    aligned = pd.Series(index=hourly_index)
    for i, date in enumerate(hourly_dates):
        if date in daily_proxy_dated.index:
            aligned.iloc[i] = daily_proxy_dated.loc[date]
    
    # Forward fill any gaps
    aligned = aligned.ffill().bfill()
    
    return aligned


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Demo: Calculate Polymarket proxy from price data."""
    print("=" * 60)
    print("POLYMARKET PROXY CALCULATOR")
    print("Estimating prediction market sentiment from price momentum")
    print("=" * 60)
    
    # Try to load SOL price data
    try:
        df = pd.read_csv('SOL_1h_data.csv')
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        prices = df['close']
        print(f"\nLoaded SOL price data: {len(prices)} candles")
    except:
        print("\nSOL_1h_data.csv not found, generating sample data...")
        # Generate sample price data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        # Random walk with drift
        returns = np.random.normal(0.001, 0.03, len(dates))
        prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
        print(f"Generated {len(prices)} daily prices")
    
    # Resample to daily if hourly
    if len(prices) > 1000:
        print("Resampling to daily data...")
        prices_daily = resample_to_daily(prices)
    else:
        prices_daily = prices
    
    print(f"Using {len(prices_daily)} data points for proxy calculation")
    
    # Calculate Polymarket proxy
    print("\nCalculating Polymarket Proxy...")
    proxy_df = calculate_polymarket_proxy(prices_daily)
    
    # Display results
    print("\n" + "=" * 60)
    print("POLYMARKET PROXY RESULTS")
    print("=" * 60)
    
    print("\nRecent Values:")
    display_cols = ['price', 'roc_score', 'momentum_rsi', 'ma_trend_score', 'polymarket_proxy']
    print(proxy_df[display_cols].tail(10).round(2).to_string())
    
    # Current signal
    current_proxy = proxy_df['polymarket_proxy_smoothed'].iloc[-1]
    signal = calculate_polymarket_signal(current_proxy)
    
    print(f"\n{'='*60}")
    print("CURRENT SIGNAL")
    print(f"{'='*60}")
    print(f"Polymarket Proxy Score: {current_proxy:.1f}/100")
    print(f"Signal: {signal['signal']}")
    print(f"Description: {signal['description']}")
    print(f"Action: {signal['action']}")
    
    # Validate correlation with forward returns
    print(f"\n{'='*60}")
    print("PROXY VALIDATION")
    print(f"{'='*60}")
    correlations = validate_proxy_correlation(prices_daily)
    print("\nCorrelations with future returns:")
    for key, value in correlations.items():
        print(f"  {key}: {value:.4f}")
    
    # Statistics
    print(f"\n{'='*60}")
    print("PROXY STATISTICS")
    print(f"{'='*60}")
    print(f"Mean: {proxy_df['polymarket_proxy'].mean():.1f}")
    print(f"Std: {proxy_df['polymarket_proxy'].std():.1f}")
    print(f"Min: {proxy_df['polymarket_proxy'].min():.1f}")
    print(f"Max: {proxy_df['polymarket_proxy'].max():.1f}")
    
    # Distribution
    bullish = (proxy_df['polymarket_proxy'] > 55).sum()
    bearish = (proxy_df['polymarket_proxy'] < 45).sum()
    neutral = len(proxy_df) - bullish - bearish
    
    print(f"\nSignal Distribution:")
    print(f"  Bullish (>55): {bullish} ({bullish/len(proxy_df)*100:.1f}%)")
    print(f"  Neutral (45-55): {neutral} ({neutral/len(proxy_df)*100:.1f}%)")
    print(f"  Bearish (<45): {bearish} ({bearish/len(proxy_df)*100:.1f}%)")
    
    print("\nDone!")
    
    return proxy_df


if __name__ == "__main__":
    proxy_df = main()
