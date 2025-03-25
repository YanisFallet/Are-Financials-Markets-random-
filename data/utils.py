import mplfinance as mpf

def save_candlestick_chart(df, filename="candlestick_chart.png"):
    """
    Sauvegarde un graphique en chandeliers japonais au format PNG.
    
    :param candles: Liste de bougies japonaises (dictionnaires)
    :param filename: Nom du fichier de sortie
    """
    
    # Configuration personnalisée pour supprimer les axes et la grille
    style = mpf.make_mpf_style(
        gridstyle='', 
        y_on_right=False,
        facecolor='white',
        edgecolor='white',
        figcolor='white'
    )
    
    # Configuration pour cacher les axes
    kwargs = {
        'type': 'candle',
        'style': style,
        'volume': False,
        'savefig': filename,
        'axisoff': True
    }
    
    mpf.plot(df, **kwargs)
    print(f"Graphique sauvegardé sous {filename}")