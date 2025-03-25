import random
import datetime
import math
import matplotlib.pyplot as plt


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
        low_price = min(candle_points)
        
        # Création de la bougie
        candles.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2)
        })
        
        timestamp += datetime.timedelta(minutes=dt)
    
    return candles




def plot_geo_brownian_motion(n_plots, n =100 ,drift = 0.005, sigma = 0.02, dt =1):
    plt.figure(figsize=(12,6))
    for i in range(n_plots):
        plt.plot([elem["close"] for elem in generate_candlestick_data(n = n, mu = drift, sigma = sigma, dt = dt)])
        
    plt.xlabel(f"timestamp dt = {dt}")
    plt.ylabel(f'Prices')
    plt.title(f"Geo brownian motion n = {n_plots}, drift = {drift}, vol = {sigma}")
    plt.show()
    
if __name__ == '__main__':
    # plot_geo_brownian_motion(1000)
    save_candlestick_chart(generate_candlestick_data(n = 100, mu = 0.0005, sigma = 0.02, dt = 1), "candlestick_chart.png")