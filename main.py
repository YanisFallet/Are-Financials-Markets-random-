#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Add models and data to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from project modules
from data.data_generator import create_dataset
from data.time_series_db import populate_database
from data.image_processor import get_data_loaders, preprocess_image
from CNN.model import CNNClassifier, ShallowCNN
from ViT.model import VisionTransformer, SmallViT
from ResNet.model import resnet18, resnet34, resnet50
from Time_series.models import LSTMClassifier, GRUClassifier, TransformerClassifier, TimeSeriesConvNet

class RandomWalkAnalyzer:
    """Main class for analyzing whether financial markets are truly random"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.models = {}
        self.model_paths = {
            'cnn': 'CNN/output/model.pth',
            'vit': 'ViT/output/model.pth',
            'resnet': 'ResNet/output/model.pth',
            'lstm': 'Transformers/output/lstm_model.pth',
            'gru': 'Transformers/output/gru_model.pth',
            'transformer': 'Transformers/output/transformer_model.pth'
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup_data(self, real_count=200, synthetic_count=200):
        """
        Setup the project data
        
        Args:
            real_count (int): Number of real chart images to generate
            synthetic_count (int): Number of synthetic chart images to generate
        """
        print("Setting up data...")
        
        # Create image dataset
        print(f"Creating image dataset with {real_count} real and {synthetic_count} synthetic samples...")
        real_dir, synthetic_dir = create_dataset(
            base_dir='data/samples',
            real_count=real_count,
            synthetic_count=synthetic_count
        )
        
        # Populate time series database
        print("Populating time series database...")
        populate_database(
            db_path="data/price_data.db",
            real_count=real_count,
            synthetic_count=synthetic_count
        )
        
        print("Data setup complete.")
        return real_dir, synthetic_dir
    
    def train_model(self, model_type, epochs=20, batch_size=32):
        """
        Train a specific model
        
        Args:
            model_type (str): Type of model to train (cnn, vit, resnet, lstm, gru, transformer)
            epochs (int): Number of epochs to train for
            batch_size (int): Batch size for training
        """
        print(f"Training {model_type} model...")
        
        command = []
        if model_type == 'cnn':
            command = [
                'python', 'CNN/train.py',
                '--real_dir', 'data/samples/real',
                '--synthetic_dir', 'data/samples/synthetic',
                '--batch_size', str(batch_size),
                '--num_epochs', str(epochs),
                '--model_type', 'standard'
            ]
        elif model_type == 'vit':
            command = [
                'python', 'ViT/train.py',
                '--real_dir', 'data/samples/real',
                '--synthetic_dir', 'data/samples/synthetic',
                '--batch_size', str(batch_size),
                '--num_epochs', str(epochs),
                '--model_type', 'small'
            ]
        elif model_type == 'resnet':
            command = [
                'python', 'ResNet/train.py',
                '--real_dir', 'data/samples/real',
                '--synthetic_dir', 'data/samples/synthetic',
                '--batch_size', str(batch_size),
                '--num_epochs', str(epochs),
                '--model_type', 'resnet18'
            ]
        elif model_type in ['lstm', 'gru', 'transformer', 'cnn_1d']:
            ts_model_type = 'cnn' if model_type == 'cnn_1d' else model_type
            command = [
                'python', 'Transformers/train.py',
                '--batch_size', str(batch_size),
                '--num_epochs', str(epochs),
                '--model_type', ts_model_type
            ]
        
        # Execute the training script
        if command:
            subprocess.run(command)
        else:
            print(f"Unknown model type: {model_type}")
    
    def train_all_models(self, epochs=20, batch_size=32):
        """
        Train all models
        
        Args:
            epochs (int): Number of epochs to train for
            batch_size (int): Batch size for training
        """
        # Train image-based models
        for model_type in ['cnn', 'vit', 'resnet']:
            self.train_model(model_type, epochs, batch_size)
        
        # Train time series models
        for model_type in ['lstm', 'gru', 'transformer']:
            self.train_model(model_type, epochs, batch_size)
    
    def load_model(self, model_type):
        """
        Load a trained model
        
        Args:
            model_type (str): Type of model to load
            
        Returns:
            model: The loaded model
        """
        model_path = self.model_paths.get(model_type)
        if not model_path or not os.path.exists(model_path):
            print(f"Model {model_type} not found at {model_path}")
            return None
        
        print(f"Loading {model_type} model from {model_path}...")
        
        # Create an instance of the model
        model = None
        if model_type == 'cnn':
            model = CNNClassifier(in_channels=1, num_classes=2)
        elif model_type == 'vit':
            model = SmallViT(in_channels=1, num_classes=2)
        elif model_type == 'resnet':
            model = resnet18(in_channels=1, num_classes=2)
        elif model_type == 'lstm':
            model = LSTMClassifier(input_size=4, hidden_size=128, num_layers=2)
        elif model_type == 'gru':
            model = GRUClassifier(input_size=4, hidden_size=128, num_layers=2)
        elif model_type == 'transformer':
            model = TransformerClassifier(input_size=4, d_model=128, num_layers=4)
        
        if model is not None:
            # Load model weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[model_type] = model
            return model
        else:
            print(f"Could not create model of type {model_type}")
            return None
    
    def load_all_models(self):
        """Load all trained models"""
        for model_type in self.model_paths.keys():
            self.load_model(model_type)
    
    def classify_image(self, image_path, model_type='cnn'):
        """
        Classify an image as real or synthetic
        
        Args:
            image_path (str): Path to the image to classify
            model_type (str): Type of model to use for classification
            
        Returns:
            dict: Classification results
        """
        # Load model if not already loaded
        if model_type not in self.models:
            model = self.load_model(model_type)
            if model is None:
                return None
        else:
            model = self.models[model_type]
        
        # Preprocess image
        image = preprocess_image(image_path)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Classify
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Get class and probabilities
        predicted_class = "Real" if predicted.item() == 1 else "Synthetic"
        synthetic_prob = probabilities[0, 0].item()
        real_prob = probabilities[0, 1].item()
        
        return {
            'predicted_class': predicted_class,
            'synthetic_probability': synthetic_prob,
            'real_probability': real_prob
        }
    
    def compare_model_performance(self):
        """
        Compare performance of all trained models
        
        Returns:
            dict: Performance metrics for all models
        """
        # Load all models if not already loaded
        self.load_all_models()
        
        # Results dictionary
        results = {}
        
        # Compare image-based models
        print("Comparing image-based models...")
        image_models = ['cnn', 'vit', 'resnet']
        for model_type in image_models:
            if model_type in self.models:
                # Get validation data
                train_loader, val_loader = get_data_loaders(
                    'data/samples/real',
                    'data/samples/synthetic',
                    batch_size=32
                )
                
                # Evaluate model
                results[model_type] = self._evaluate_model(self.models[model_type], val_loader)
        
        # Compare time series models - we'll skip this part for now as it requires specific data setup
        
        # Print results
        print("\nModel Performance Comparison:")
        for model_type, metrics in results.items():
            print(f"\n{model_type.upper()} Model:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print("Classification Report:")
            print(metrics['report'])
        
        return results
    
    def _evaluate_model(self, model, dataloader):
        """
        Evaluate a model on a dataset
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with validation/test data
            
        Returns:
            dict: Evaluation metrics
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # Store predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=['Synthetic', 'Real'])
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        
        return {
            'confusion_matrix': cm,
            'report': report,
            'accuracy': accuracy
        }

def main():
    parser = argparse.ArgumentParser(description='Random Walk Analyzer - Determine if financial markets are truly random')
    
    parser.add_argument('action', choices=['setup', 'train', 'evaluate', 'classify'],
                        help='Action to perform')
    parser.add_argument('--model', choices=['cnn', 'vit', 'resnet', 'lstm', 'gru', 'transformer', 'all'],
                        default='all', help='Model to use')
    parser.add_argument('--real_count', type=int, default=200,
                        help='Number of real samples to generate')
    parser.add_argument('--synthetic_count', type=int, default=200,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--image', type=str,
                        help='Path to image for classification')
    
    args = parser.parse_args()
    
    analyzer = RandomWalkAnalyzer()
    
    if args.action == 'setup':
        analyzer.setup_data(args.real_count, args.synthetic_count)
    
    elif args.action == 'train':
        if args.model == 'all':
            analyzer.train_all_models(args.epochs, args.batch_size)
        else:
            analyzer.train_model(args.model, args.epochs, args.batch_size)
    
    elif args.action == 'evaluate':
        analyzer.compare_model_performance()
    
    elif args.action == 'classify':
        if not args.image:
            print("Error: --image argument is required for 'classify' action")
            return
        
        # Classify using all models or a specific model
        if args.model == 'all':
            for model_type in ['cnn', 'vit', 'resnet']:
                result = analyzer.classify_image(args.image, model_type)
                if result:
                    print(f"\n{model_type.upper()} Classification:")
                    print(f"Predicted class: {result['predicted_class']}")
                    print(f"Probability of Synthetic: {result['synthetic_probability']:.4f}")
                    print(f"Probability of Real: {result['real_probability']:.4f}")
        else:
            result = analyzer.classify_image(args.image, args.model)
            if result:
                print(f"\n{args.model.upper()} Classification:")
                print(f"Predicted class: {result['predicted_class']}")
                print(f"Probability of Synthetic: {result['synthetic_probability']:.4f}")
                print(f"Probability of Real: {result['real_probability']:.4f}")

if __name__ == '__main__':
    main()
