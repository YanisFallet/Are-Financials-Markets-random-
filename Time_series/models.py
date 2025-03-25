import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    """LSTM model for time series classification"""
    
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.2, num_classes=2):
        """
        Initialize LSTM model
        
        Args:
            input_size (int): Number of features per time step (4 for OHLC data)
            hidden_size (int): Number of hidden units
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
            num_classes (int): Number of output classes
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        # output: (batch_size, seq_len, hidden_size * 2) - bidirectional
        output, _ = self.lstm(x)
        
        # Apply attention
        attention_weights = self.attention(output)  # (batch_size, seq_len, 1)
        context_vector = torch.sum(attention_weights * output, dim=1)  # (batch_size, hidden_size * 2)
        
        # Classification
        logits = self.fc(context_vector)
        
        return logits

class GRUClassifier(nn.Module):
    """GRU model for time series classification"""
    
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.2, num_classes=2):
        """
        Initialize GRU model
        
        Args:
            input_size (int): Number of features per time step (4 for OHLC data)
            hidden_size (int): Number of hidden units
            num_layers (int): Number of GRU layers
            dropout (float): Dropout probability
            num_classes (int): Number of output classes
        """
        super(GRUClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # GRU forward pass
        # output: (batch_size, seq_len, hidden_size * 2) - bidirectional
        _, h_n = self.gru(x)  # h_n: (2 * num_layers, batch_size, hidden_size)
        
        # Get the final states of the forward and backward pass
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)  # (num_layers, 2, batch_size, hidden_size)
        h_n = h_n[-1]  # (2, batch_size, hidden_size) - last layer
        h_n = h_n.permute(1, 0, 2)  # (batch_size, 2, hidden_size)
        h_n = h_n.reshape(-1, 2 * self.hidden_size)  # (batch_size, 2 * hidden_size)
        
        # Classification
        logits = self.fc(h_n)
        
        return logits

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding
        
        Args:
            d_model (int): Embedding dimension
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register buffer (not a parameter, but should be saved and moved with model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]

class TransformerClassifier(nn.Module):
    """Transformer model for time series classification"""
    
    def __init__(self, input_size=4, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, num_classes=2, max_len=5000):
        """
        Initialize Transformer model
        
        Args:
            input_size (int): Number of features per time step (4 for OHLC data)
            d_model (int): Embedding dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout probability
            num_classes (int): Number of output classes
            max_len (int): Maximum sequence length
        """
        super(TransformerClassifier, self).__init__()
        
        # Feature projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Project input features to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer forward pass
        # output: (batch_size, seq_len, d_model)
        output = self.transformer_encoder(x)
        
        # Global average pooling
        # output: (batch_size, d_model)
        output = output.mean(dim=1)
        
        # Classification
        logits = self.fc(output)
        
        return logits

class TimeSeriesConvNet(nn.Module):
    """1D CNN model for time series classification"""
    
    def __init__(self, input_size=4, num_filters=64, kernel_size=3, dropout=0.2, num_classes=2):
        """
        Initialize 1D CNN model
        
        Args:
            input_size (int): Number of features per time step (4 for OHLC data)
            num_filters (int): Number of filters in convolutional layers
            kernel_size (int): Size of the convolutional kernel
            dropout (float): Dropout probability
            num_classes (int): Number of output classes
        """
        super(TimeSeriesConvNet, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding=(kernel_size-1)//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size, padding=(kernel_size-1)//2)
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.conv3 = nn.Conv1d(num_filters*2, num_filters*4, kernel_size, padding=(kernel_size-1)//2)
        self.bn3 = nn.BatchNorm1d(num_filters*4)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.fc = nn.Linear(num_filters*4, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Transpose to channel-first format for 1D convolution
        # (batch_size, seq_len, input_size) -> (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        
        # Classification
        x = self.fc(x)
        
        return x 