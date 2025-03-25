# Random Walk Analyzer

A research project to determine whether financial markets are truly random by using deep learning to distinguish between real market data and synthetic data generated using geometric Brownian motion.

## Project Overview

This project implements two main approaches to analyze financial market randomness:

1. **Image-Based Classification**: Converting price charts to images and using computer vision models to classify them as real or synthetic.
   - CNN architecture
   - Vision Transformer (ViT)
   - ResNet architecture

2. **Time Series Classification**: Analyzing raw OHLC price data using sequence models.
   - LSTM
   - GRU
   - Transformer architecture

The hypothesis is that if markets are truly random and follow geometric Brownian motion, then deep learning models should struggle to distinguish between real market data and synthetic data (accuracy close to 50%). If the models achieve significantly higher accuracy, it suggests that real markets have patterns or structure not present in pure random walks.

## Project Structure

```
📦 random_walk/
 ┣ 📂 CNN/                   # CNN model implementation
 ┃ ┣ 📜 model.py            # CNN architecture
 ┃ ┗ 📜 train.py            # CNN training script
 ┣ 📂 ResNet/                # ResNet implementation
 ┃ ┣ 📜 model.py            # ResNet architecture
 ┃ ┗ 📜 train.py            # ResNet training script
 ┣ 📂 ViT/                   # Vision Transformer implementation
 ┃ ┣ 📜 model.py            # Vision Transformer architecture
 ┃ ┗ 📜 train.py            # Vision Transformer training script
 ┣ 📂 Transformers/          # Time series models
 ┃ ┣ 📜 models.py           # LSTM, GRU, Transformer implementations
 ┃ ┗ 📜 train.py            # Training script for time series models
 ┣ 📂 data/                  # Data processing modules
 ┃ ┣ 📜 data_generator.py   # Generate dataset of real and synthetic charts
 ┃ ┣ 📜 generate_fake_prices.py # Generate synthetic price data
 ┃ ┣ 📜 image_processor.py  # Process chart images for model input
 ┃ ┣ 📜 retrieve_data.py    # Fetch real market data from API
 ┃ ┣ 📜 time_series_db.py   # Database for time series data
 ┃ ┗ 📜 utils.py            # Utilities for data processing
 ┣ 📜 main.py                # Main script to tie everything together
 ┣ 📜 requirements.txt       # Project dependencies
 ┗ 📜 README.md              # Project documentation
```

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd random_walk
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your Binance API credentials (for fetching real market data):
   ```
   API_KEY_BINANCE=your_api_key
   API_SECRET_BINANCE=your_api_secret
   ```

## Usage

The project can be used through the `main.py` script which provides a unified interface for all functionality:

### 1. Setup Data

Generate both image and time series datasets:

```
python main.py setup --real_count 200 --synthetic_count 200
```

### 2. Train Models

Train all models:

```
python main.py train --model all --epochs 20 --batch_size 32
```

Or train a specific model:

```
python main.py train --model cnn --epochs 20 --batch_size 32
```

Available model options: `cnn`, `vit`, `resnet`, `lstm`, `gru`, `transformer`

### 3. Evaluate Models

Compare performance of all trained models:

```
python main.py evaluate
```

### 4. Classify New Images

Classify a chart image using a trained model:

```
python main.py classify --model cnn --image path/to/chart.png
```

## Research Methodology

1. **Data Collection**: Real market data is collected from Binance API for multiple cryptocurrencies (BTC, ETH, SOL) across various timeframes.

2. **Synthetic Data Generation**: Random walk price data is generated using geometric Brownian motion, which is commonly used to model financial markets.

3. **Image Processing**: Both real and synthetic price charts are converted to candlestick chart images, then preprocessed (grayscale, resizing, normalization).

4. **Model Training**: Multiple model architectures are trained to distinguish between real and synthetic data.

5. **Performance Analysis**: Model performance is analyzed to determine if they can reliably distinguish real from synthetic data.

## Additional Notes

- The project requires a Binance API key to fetch real cryptocurrency data.
- Model training can be computationally intensive, especially for the Vision Transformer and Transformer models.
- For best results, generate a large and diverse dataset of both real and synthetic charts.

## License

[MIT License](LICENSE) 