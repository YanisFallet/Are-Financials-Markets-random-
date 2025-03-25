import os
import pandas as pd
import mplfinance as mpf
import requests
import dotenv
import hmac
import hashlib
from datetime import datetime, timedelta
import time

from utils import save_candlestick_chart

# Charger les variables d'environnement
dotenv.load_dotenv()

# Récupérer les clés API de Binance
API_KEY = os.getenv('API_KEY_BINANCE')
API_SECRET = os.getenv('API_SECRET_BINANCE')

# URL de base de l'API Binance
BASE_URL = 'https://api.binance.com'

def get_server_time():
    """Récupère le temps serveur de Binance"""
    endpoint = f"{BASE_URL}/api/v3/time"
    response = requests.get(endpoint)
    return response.json()['serverTime']

def get_signature(query_string):
    """Génère une signature HMAC SHA256 pour l'authentification"""
    return hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def get_klines(symbol='BTCUSDT', interval='30m', limit=50):
    """Récupère les données de chandeliers (klines) pour un symbole donné"""
    endpoint = f"{BASE_URL}/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(endpoint, params=params)
    return response.json()


def klines_to_dataframe(klines_data):
    """Convertit les données de chandeliers en DataFrame pandas"""
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines_data, columns=columns)
    
    # Convertir les types de données
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].astype(float)
    
    # Convertir le timestamp en datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

def plot_klines(df):
    """Affiche les données de chandeliers avec mplfinance"""
    mpf.plot(df, type='candle', style='charles', title='BTCUSDT 1m', ylabel='Prix')

def get_historical_klines(symbol, interval, start_time=None, end_time=None, limit=1000):
    """
    Récupère les données historiques de klines depuis Binance.
    
    :param symbol: Le symbole de trading (ex: 'BTCUSDT')
    :param interval: L'intervalle de temps ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
    :param start_time: Timestamp de début (en millisecondes) ou datetime object
    :param end_time: Timestamp de fin (en millisecondes) ou datetime object
    :param limit: Nombre maximum de klines par requête (max 1000)
    :return: DataFrame contenant les données historiques
    """
    endpoint = 'https://api3.binance.com/api/v3/klines'
    
    # Conversion des dates en timestamps si nécessaire
    if isinstance(start_time, datetime):
        start_time = int(start_time.timestamp() * 1000)
    if isinstance(end_time, datetime):
        end_time = int(end_time.timestamp() * 1000)
    
    all_klines = []
    current_start = start_time
    
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if current_start:
            params['startTime'] = current_start
        if end_time:
            params['endTime'] = end_time
            
        headers = {'X-MBX-APIKEY': API_KEY}
        
        try:
            response = requests.get(endpoint, params=params, headers=headers)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:  # Si pas de données, on arrête
                break
                
            all_klines.extend(klines)
            
            # Si on a moins de klines que la limite, on a atteint la fin
            if len(klines) < limit:
                break
                
            # Mise à jour du timestamp de début pour la prochaine requête
            current_start = klines[-1][0] + 1
            
            # Pause pour respecter les limites de rate
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération des données: {e}")
            break
    
    # Conversion en DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Conversion des types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df.index = pd.to_datetime(df['timestamp'])
    df = df[['open', 'high', 'low', 'close']]    
    return df

if __name__ == '__main__':
    # Récupérer le temps serveur de Binance
    server_time = get_server_time()
    print(f"Temps serveur de Binance: {server_time}")

    # Récupérer les données des 7 derniers jours
    end_time = datetime.now()
    start_time = end_time - timedelta(days=3)
    
    df = get_historical_klines(
        symbol='BTCUSDT',
        interval='1h',
        start_time=start_time,
        end_time=end_time
    )
    print(df.head())
    
    save_candlestick_chart(df, filename=f"BTCUSDT_1h_{start_time.strftime('%Y-%m-%d')}_{end_time.strftime('%Y-%m-%d')}.png")