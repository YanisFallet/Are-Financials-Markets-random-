import os
import pandas as pd
import numpy as np
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from retrieve_data import get_historical_klines
from generate_fake_prices import generate_candlestick_data
from datetime import datetime, timedelta

class TimeSeriesDatabase:
    """Class to manage a database of time series price data"""
    
    def __init__(self, db_path="data/price_data.db"):
        """
        Initialize the database
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.ensure_db_exists()
        
    def ensure_db_exists(self):
        """Create the database and tables if they don't exist"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(self.db_path)
        c = self.conn.cursor()
        
        # Create tables
        c.execute('''
        CREATE TABLE IF NOT EXISTS real_data (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            timestamp TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            interval TEXT
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS synthetic_data (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            mu REAL,
            sigma REAL,
            dt INTEGER
        )
        ''')
        
        self.conn.commit()
    
    def add_real_data(self, df, symbol, interval):
        """
        Add real market data to the database
        
        Args:
            df (DataFrame): DataFrame with OHLC data
            symbol (str): Trading symbol
            interval (str): Time interval
        """
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        
        # Convert DataFrame to list of tuples
        data = []
        for idx, row in df.iterrows():
            data.append((
                None,  # id (auto-incremented)
                symbol,
                idx.strftime('%Y-%m-%d %H:%M:%S'),
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                interval
            ))
        
        # Insert data
        c = self.conn.cursor()
        c.executemany(
            'INSERT INTO real_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            data
        )
        self.conn.commit()
    
    def add_synthetic_data(self, df, mu, sigma, dt):
        """
        Add synthetic market data to the database
        
        Args:
            df (DataFrame): DataFrame with OHLC data
            mu (float): Drift parameter used to generate data
            sigma (float): Volatility parameter used to generate data
            dt (int): Time step used to generate data
        """
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        
        # Convert DataFrame to list of tuples
        data = []
        for idx, row in df.iterrows():
            data.append((
                None,  # id (auto-incremented)
                idx.strftime('%Y-%m-%d %H:%M:%S'),
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                mu,
                sigma,
                dt
            ))
        
        # Insert data
        c = self.conn.cursor()
        c.executemany(
            'INSERT INTO synthetic_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            data
        )
        self.conn.commit()
    
    def fetch_real_data(self, symbol=None, interval=None, limit=None):
        """
        Fetch real market data from the database
        
        Args:
            symbol (str): Trading symbol to filter by
            interval (str): Time interval to filter by
            limit (int): Maximum number of records to return
            
        Returns:
            DataFrame: DataFrame with OHLC data
        """
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        
        query = 'SELECT * FROM real_data'
        params = []
        
        # Add filters
        if symbol or interval:
            query += ' WHERE'
            
            if symbol:
                query += ' symbol = ?'
                params.append(symbol)
                
                if interval:
                    query += ' AND'
            
            if interval:
                query += ' interval = ?'
                params.append(interval)
        
        # Add limit
        if limit:
            query += f' LIMIT {limit}'
        
        # Execute query
        df = pd.read_sql_query(query, self.conn, params=params)
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def fetch_synthetic_data(self, limit=None):
        """
        Fetch synthetic market data from the database
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            DataFrame: DataFrame with OHLC data
        """
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        
        query = 'SELECT * FROM synthetic_data'
        
        # Add limit
        if limit:
            query += f' LIMIT {limit}'
        
        # Execute query
        df = pd.read_sql_query(query, self.conn)
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def close(self):
        """Close the database connection"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

class TimeSeriesDataset(Dataset):
    """Dataset for time series price data"""
    
    def __init__(self, real_data, synthetic_data, sequence_length=50):
        """
        Initialize the dataset
        
        Args:
            real_data (DataFrame): DataFrame with real OHLC data
            synthetic_data (DataFrame): DataFrame with synthetic OHLC data
            sequence_length (int): Length of each sequence
        """
        self.sequence_length = sequence_length
        
        # Prepare data
        X_real, y_real = self._prepare_sequences(real_data, label=1)
        X_synthetic, y_synthetic = self._prepare_sequences(synthetic_data, label=0)
        
        # Combine data
        self.X = torch.cat([X_real, X_synthetic], dim=0)
        self.y = torch.cat([y_real, y_synthetic], dim=0)
        
    def _prepare_sequences(self, df, label):
        """
        Prepare sequences from DataFrame
        
        Args:
            df (DataFrame): DataFrame with OHLC data
            label (int): Label for the sequences (0 for synthetic, 1 for real)
            
        Returns:
            tuple: (X, y) tensors
        """
        # Extract OHLC data
        data = df[['open', 'high', 'low', 'close']].values
        
        # Normalize data (per sequence)
        sequences = []
        
        for i in range(len(data) - self.sequence_length + 1):
            # Extract sequence
            sequence = data[i:i+self.sequence_length]
            
            # Normalize sequence
            scaler = MinMaxScaler()
            sequence = scaler.fit_transform(sequence)
            
            sequences.append(sequence)
        
        # Convert to tensors
        X = torch.tensor(np.array(sequences), dtype=torch.float32)
        y = torch.tensor([label] * len(sequences), dtype=torch.long)
        
        return X, y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def populate_database(db_path="data/price_data.db", real_count=5000, synthetic_count=5000):
    """
    Populate the database with real and synthetic data
    
    Args:
        db_path (str): Path to the database file
        real_count (int): Number of real data points to add
        synthetic_count (int): Number of synthetic data points to add
        
    Returns:
        TimeSeriesDatabase: The populated database
    """
    db = TimeSeriesDatabase(db_path)
    
    # Add real data
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    intervals = ['15m', '1h', '4h', '1d']
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    print(f"Fetching real data for database...")
    
    for symbol in symbols:
        for interval in intervals:
            try:
                df = get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time
                )
                
                db.add_real_data(df, symbol, interval)
                print(f"Added {len(df)} {interval} records for {symbol}")
                
            except Exception as e:
                print(f"Error fetching data for {symbol} {interval}: {e}")
    
    # Add synthetic data
    print(f"Generating synthetic data for database...")
    
    # Generate synthetic data in batches
    batch_size = 100
    batches = (synthetic_count + batch_size - 1) // batch_size
    
    for _ in range(batches):
        # Generate random parameters
        n = np.random.randint(50, 200)
        start_price = np.random.uniform(100, 10000)
        mu = np.random.uniform(-0.001, 0.001)
        sigma = np.random.uniform(0.005, 0.03)
        dt = np.random.choice([1, 5, 15, 60])
        
        # Generate data
        df, _ = generate_candlestick_data(
            n=n, 
            start_price=start_price,
            mu=mu,
            sigma=sigma,
            dt=dt
        )
        
        # Add to database
        db.add_synthetic_data(df, mu, sigma, dt)
        print(f"Added {len(df)} synthetic records with mu={mu:.4f}, sigma={sigma:.4f}")
    
    print(f"Database populated at {db_path}")
    
    return db

def get_time_series_loaders(db_path="data/price_data.db", sequence_length=50, batch_size=32, train_ratio=0.8):
    """
    Create train and validation data loaders for time series data
    
    Args:
        db_path (str): Path to the database file
        sequence_length (int): Length of each sequence
        batch_size (int): Batch size for the data loaders
        train_ratio (float): Ratio of data to use for training
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Open database
    db = TimeSeriesDatabase(db_path)
    
    # Fetch data
    real_data = db.fetch_real_data()
    synthetic_data = db.fetch_synthetic_data()
    
    # Close database
    db.close()
    
    # Create dataset
    dataset = TimeSeriesDataset(real_data, synthetic_data, sequence_length)
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == '__main__':
    # Populate database with sample data
    db = populate_database(real_count=1000, synthetic_count=1000)
    
    # Create data loaders
    train_loader, val_loader = get_time_series_loaders(sequence_length=50, batch_size=32)
    
    # Print dataset info
    for X, y in train_loader:
        print(f"Batch shape: {X.shape}, Label shape: {y.shape}")
        print(f"X min: {X.min().item()}, X max: {X.max().item()}")
        print(f"Label distribution: {y.sum().item()} real, {len(y) - y.sum().item()} synthetic")
        break 