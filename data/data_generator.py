import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import random
from tqdm import tqdm

from retrieve_data import get_historical_klines
from generate_fake_prices import generate_candlestick_data
from utils import save_candlestick_chart

def ensure_dir(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_real_charts(output_dir, symbols=None, intervals=None, days_back=30, samples_per_symbol=20):
    """
    Generate chart images from real cryptocurrency data
    
    Args:
        output_dir (str): Directory to save chart images
        symbols (list): List of cryptocurrency symbols to use
        intervals (list): List of time intervals
        days_back (int): Number of days to look back for data
        samples_per_symbol (int): Number of samples to generate per symbol/interval combination
    """
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 'DOTUSDT']
    
    if intervals is None:
        intervals = ['1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
    
    ensure_dir(output_dir)
    
    print(f"Generating real chart images for {symbols} with intervals {intervals}...")
    
    for symbol in symbols:
        for interval in intervals:
            try:
                # Generate random end_time within the last year
                max_end_time = datetime.now()
                random_days_back = random.randint(0, 1000)  # Random days back for end_time
                end_time = max_end_time - timedelta(days=random_days_back)
                
                # Generate random start_time based on days_back from the random end_time
                start_time = end_time - timedelta(days=days_back)
                
                # Get historical data
                df = get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if len(df) < 10:  # Skip if not enough data
                    continue
                    
                # Generate random samples from different time periods
                for _ in range(samples_per_symbol):
                    # Random window size for diverse chart patterns
                    window_size = random.randint(10, min(500, len(df) - 1))
                    
                    # Random starting point
                    start_idx = random.randint(0, len(df) - window_size - 1)
                    
                    # Extract random window
                    window_df = df.iloc[start_idx:start_idx+window_size]
                    
                    # Save chart image
                    timestamp = window_df.index[-1].strftime('%Y%m%d%H%M%S')
                    filename = os.path.join(output_dir, f"chart_{timestamp}.png")
                    save_candlestick_chart(window_df, filename=filename)
                    
                    # Small delay to avoid API rate limits
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error generating data for {symbol} {interval}: {e}")
    
    print(f"Real chart generation complete. Images saved to {output_dir}")

def generate_synthetic_charts(output_dir, count=1000, timeframes=None):
    """
    Generate synthetic chart images using geometric Brownian motion
    
    Args:
        output_dir (str): Directory to save chart images
        count (int): Number of synthetic charts to generate
        timeframes (list): List of timeframes (candle counts) to generate
    """
    if timeframes is None:
        timeframes = [20, 50, 100, 200]
    
    ensure_dir(output_dir)
    
    print(f"Generating {count} synthetic chart images...")
    
    start_price = 100
    mu = random.uniform(-0.001, 0.001)
    sigma = random.uniform(0.005, 0.03)
    
    for i in tqdm(range(count)):
        # Randomly select parameters
        n = random.choice(timeframes)
        mu = random.uniform(-0.001, 0.001)  # Drift parameter
        sigma = random.uniform(0.005, 0.03)  # Volatility parameter
        dt = random.choice([1, 5, 15, 60])  # Time interval in minutes
        
        # Generate synthetic data
        df, _ = generate_candlestick_data(n=n, start_price=start_price, mu=mu, sigma=sigma, dt=dt)
        
        # Save chart image
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = os.path.join(output_dir, f"synthetic_n{n}_mu{mu:.4f}_sigma{sigma:.4f}_{timestamp}_{i}.png")
        save_candlestick_chart(df, filename=filename)
        
        # Small delay to avoid resource contention
        time.sleep(0.05)
    
    print(f"Synthetic chart generation complete. Images saved to {output_dir}")

def create_dataset(base_dir='data/samples', real_count=200, synthetic_count=200):
    """
    Create a complete dataset with both real and synthetic charts
    
    Args:
        base_dir (str): Base directory for the dataset
        real_count (int): Number of real chart images to generate
        synthetic_count (int): Number of synthetic chart images to generate
    
    Returns:
        tuple: (real_dir, synthetic_dir) - paths to the generated data directories
    """
    # Create directories
    real_dir = os.path.join(base_dir, 'real')
    synthetic_dir = os.path.join(base_dir, 'synthetic')
    
    ensure_dir(base_dir)
    ensure_dir(real_dir)
    ensure_dir(synthetic_dir)
    
    # Calculate samples per symbol/interval
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    intervals = ['30m', '1h', '4h', '1d']
    samples_per_combo = max(1, real_count // (len(symbols) * len(intervals)))
    
    # Generate real charts
    generate_real_charts(
        output_dir=real_dir,
        symbols=symbols,
        intervals=intervals,
        days_back=30,
        samples_per_symbol=samples_per_combo
    )
    
    # Generate synthetic charts
    generate_synthetic_charts(
        output_dir=synthetic_dir,
        count=synthetic_count
    )
    
    return real_dir, synthetic_dir

if __name__ == '__main__':
    # real_dir, synthetic_dir = create_dataset(real_count=100, synthetic_count=100)
    # print(f"Dataset created at:\nReal charts: {real_dir}\nSynthetic charts: {synthetic_dir}")
    generate_real_charts(output_dir='data/samples/real', symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], intervals=['1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w'], days_back=30, samples_per_symbol=20)