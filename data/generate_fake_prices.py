import random
import datetime
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import save_candlestick_chart

def generate_candlestick_data(n=10, start_price=100, mu=0.0005, sigma=0.02, dt=1):
    """
    Génère une liste de bougies japonaises en simulant un marché réaliste.
    Intègre un mouvement brownien géométrique avec dérive, une volatilité conditionnelle,
    et des cycles journaliers.
    
    :param n: Nombre de bougies à générer
    :param start_price: Prix initial
    :param mu: Taux de dérive (rendement moyen attendu par unité de temps)
    :param sigma: Volatilité de base du prix
    :param dt: Pas de temps en minutes
    :return: Liste de dictionnaires représentant les bougies
    """
    # Initialisation
    candles = []
    timestamp = datetime.datetime.now()
    
    # Paramètres pour la génération des points de prix
    points_per_candle = 10
    time_step = dt / points_per_candle
    
    # Générer tous les points de prix en continu
    all_price_points = []
    temp_price = start_price
    
    # Simulation du mouvement brownien géométrique
    for _ in range(n * points_per_candle):
        # Incrément brownien
        dW = random.gauss(0, math.sqrt(time_step))
        # Formule du mouvement brownien géométrique
        temp_price = temp_price * math.exp((mu - 0.5 * sigma**2) * time_step + sigma * dW)
        all_price_points.append(temp_price)
    
    # Création des bougies à partir des points de prix
    for i in range(n):
        start_idx = i * points_per_candle
        end_idx = (i + 1) * points_per_candle
        candle_points = all_price_points[start_idx:end_idx]
        
        # Déterminer les prix de la bougie
        if i == 0:
            open_price = candle_points[0]
        else:
            open_price = all_price_points[start_idx - 1]  # Le prix de fermeture précédent
            
        close_price = candle_points[-1]
        high_price = max(candle_points)
        low_price = min(candle_points + [open_price])
        
        n_ = np.argmin(candle_points)
        
        if low_price > open_price:
            print(f"low_price > open_price at {timestamp.strftime('%Y-%m-%d %H:%M:%S')} {n_}")
        
        # Création de la bougie
        candles.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2)
        })
        
        timestamp += datetime.timedelta(minutes=dt)
    
    return pd.DataFrame(candles, index=pd.to_datetime([elem["timestamp"] for elem in candles]))


def generate_double_heston_candlestick_data(n=10, start_price=100, mu=0.005, 
                                            v0_1=0.002, theta_1=0.02, kappa_1=1.5, sigma_1=0.2,
                                            v0_2=0.004, theta_2=0.04, kappa_2=0.3, sigma_2=0.1,
                                            rho=-0.5, dt=1):
    """
    Simule des bougies japonaises avec un modèle Double Heston
    (volatilité stochastique à deux échelles de temps : rapide et lente).
    
    :param n: Nombre de bougies à générer
    :param start_price: Prix initial
    :param mu: Taux de dérive
    :param v0_1, theta_1, kappa_1, sigma_1: Paramètres de la volatilité rapide (Heston 1)
    :param v0_2, theta_2, kappa_2, sigma_2: Paramètres de la volatilité lente (Heston 2)
    :param rho: Corrélation entre le mouvement brownien du prix et celui de la volatilité
    :param dt: Pas de temps en minutes
    :return: DataFrame des bougies japonaises et liste des prix simulés
    """
    # Initialisation
    candles = []
    timestamp = datetime.datetime.now()
    
    points_per_candle = 10
    time_step = dt / points_per_candle
    
    # Processus de volatilité stochastique
    v1 = v0_1
    v2 = v0_2
    temp_price = start_price
    all_price_points = []
    
    for _ in range(n * points_per_candle):
        # Génération des chocs brownien corrélés
        dW1 = random.gauss(0, math.sqrt(time_step))
        dW2 = rho * dW1 + math.sqrt(1 - rho**2) * random.gauss(0, math.sqrt(time_step))
        dW3 = random.gauss(0, math.sqrt(time_step))
        
        # Évolution de la volatilité rapide
        v1 = abs(v1 + kappa_1 * (theta_1 - v1) * time_step + sigma_1 * math.sqrt(v1) * dW2)
        
        # Évolution de la volatilité lente
        v2 = abs(v2 + kappa_2 * (theta_2 - v2) * time_step + sigma_2 * math.sqrt(v2) * dW3)
        
        # Volatilité totale
        sigma_eff = math.sqrt(v1 + v2)
        
        # Évolution du prix sous Double Heston
        temp_price *= math.exp((mu - 0.5 * sigma_eff**2) * time_step + sigma_eff * dW1)
        all_price_points.append(temp_price)
    
    # Construction des bougies
    for i in range(n):
        start_idx = i * points_per_candle
        end_idx = (i + 1) * points_per_candle
        candle_points = all_price_points[start_idx:end_idx]
        
        open_price = all_price_points[start_idx - 1] if i > 0 else candle_points[0]
        close_price = candle_points[-1]
        high_price = max(candle_points)
        low_price = min(candle_points + [open_price])
        
        candles.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2)
        })
        
        timestamp += datetime.timedelta(minutes=dt)
    
    return pd.DataFrame(candles, index=pd.to_datetime([elem["timestamp"] for elem in candles]))





def plot_geo_brownian_motion(n_plots, n =100 ,drift = 0.005, sigma = 0.002, dt =1):
    plt.figure(figsize=(12,6))
    for i in range(n_plots):
        plt.plot([elem["close"] for elem in generate_candlestick_data(n = n, mu = drift, sigma = sigma, dt = dt)])
        
    plt.xlabel(f"timestamp dt = {dt}")
    plt.ylabel(f'Prices')
    plt.title(f"Geo brownian motion n = {n_plots}, drift = {drift}, vol = {sigma}")
    plt.show()
    
if __name__ == '__main__':
    df = generate_double_heston_candlestick_data(n =100)
    save_candlestick_chart(df, "candlestick_chart.png")