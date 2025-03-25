import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add parent directory to path to import from data folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.time_series_db import get_time_series_loaders, populate_database
from models import LSTMClassifier, GRUClassifier, TransformerClassifier, TimeSeriesConvNet

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30):
    """
    Train the time series model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on (cuda or cpu)
        num_epochs: Number of epochs to train for
        
    Returns:
        dict: Training history
    """
    # Track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Store metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_train_loss:.4f} | '
              f'Train Acc: {epoch_train_acc:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | '
              f'Val Acc: {epoch_val_acc:.4f}')
    
    return history

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on test data
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on (cuda or cpu)
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Track metrics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=['Synthetic', 'Real'])
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'confusion_matrix': cm,
        'classification_report': report
    }

def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot to
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss curves')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy curves')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot to
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Synthetic', 'Real'],
                yticklabels=['Synthetic', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train time series model for financial data classification')
    
    parser.add_argument('--db_path', type=str, default='data/price_data.db',
                        help='Path to the SQLite database')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--model_type', type=str, default='lstm',
                        choices=['lstm', 'gru', 'transformer', 'cnn'],
                        help='Model type to use')
    parser.add_argument('--sequence_length', type=int, default=50,
                        help='Sequence length for time series data')
    parser.add_argument('--output_dir', type=str, default='Transformers/output',
                        help='Directory to save output to')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size for recurrent models')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers for models')
    parser.add_argument('--populate_db', action='store_true',
                        help='Populate the database before training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Populate database if requested
    if args.populate_db:
        print("Populating database...")
        populate_database(db_path=args.db_path, real_count=1000, synthetic_count=1000)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_time_series_loaders(
        db_path=args.db_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size
    )
    print(f"Train loader size: {len(train_loader.dataset)}")
    print(f"Validation loader size: {len(val_loader.dataset)}")
    
    # Create model
    print(f"Creating {args.model_type} model...")
    if args.model_type == 'lstm':
        model = LSTMClassifier(
            input_size=4,  # OHLC data
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        )
    elif args.model_type == 'gru':
        model = GRUClassifier(
            input_size=4,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        )
    elif args.model_type == 'transformer':
        model = TransformerClassifier(
            input_size=4,
            d_model=args.hidden_size,
            num_layers=args.num_layers
        )
    else:  # CNN
        model = TimeSeriesConvNet(
            input_size=4,
            num_filters=args.hidden_size // 2
        )
    model.to(device)
    
    # Set up optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # Train model
    print("Training model...")
    history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=args.num_epochs
    )
    
    # Evaluate model
    print("Evaluating model...")
    eval_metrics = evaluate(model, val_loader, criterion, device)
    
    print(f"Test Loss: {eval_metrics['test_loss']:.4f}")
    print(f"Test Accuracy: {eval_metrics['test_acc']:.4f}")
    print("\nClassification Report:")
    print(eval_metrics['classification_report'])
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(args.output_dir, f'{args.model_type}_training_history.png'))
    
    # Plot confusion matrix
    plot_confusion_matrix(eval_metrics['confusion_matrix'], save_path=os.path.join(args.output_dir, f'{args.model_type}_confusion_matrix.png'))
    
    # Save model
    model_path = os.path.join(args.output_dir, f'{args.model_type}_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main() 