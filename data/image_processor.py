import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def convert_to_grayscale(image_path):
    """Convert an image to grayscale"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def resize_image(image, target_size=(224, 224)):
    """Resize an image to target dimensions"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def crop_image(image, crop_percent=0.8):
    """Crop an image to remove potential chart borders/legends"""
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    crop_width = int(width * crop_percent)
    crop_height = int(height * crop_percent)
    
    start_x = center_x - crop_width // 2
    start_y = center_y - crop_height // 2
    
    cropped_image = image[start_y:start_y+crop_height, start_x:start_x+crop_width]
    return cropped_image

def normalize_image(image):
    """Normalize image pixel values to range [0, 1]"""
    return image / 255.0

def preprocess_image(image_path, target_size=(224, 224), crop_percent=0.8):
    """Complete image preprocessing pipeline"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped = crop_image(gray, crop_percent)
    resized = resize_image(cropped, target_size)
    normalized = normalize_image(resized)
    return normalized

class ChartDataset(Dataset):
    """Dataset for financial chart images"""
    def __init__(self, real_dir, synthetic_dir, transform=None):
        """
        Initialize the dataset
        
        Args:
            real_dir (str): Directory containing real chart images
            synthetic_dir (str): Directory containing synthetic chart images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.transform = transform
        
        # List all image files and create labels
        self.image_paths = []
        self.labels = []
        
        # Real charts (label 1)
        if os.path.exists(real_dir):
            real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.image_paths.extend(real_images)
            self.labels.extend([1] * len(real_images))
        
        # Synthetic charts (label 0)
        if os.path.exists(synthetic_dir):
            synthetic_images = [os.path.join(synthetic_dir, f) for f in os.listdir(synthetic_dir) 
                               if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.image_paths.extend(synthetic_images)
            self.labels.extend([0] * len(synthetic_images))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = preprocess_image(img_path)
        
        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(real_dir, synthetic_dir, batch_size=32, train_ratio=0.8):
    """
    Create train and validation data loaders
    
    Args:
        real_dir (str): Directory containing real chart images
        synthetic_dir (str): Directory containing synthetic chart images
        batch_size (int): Batch size for the data loaders
        train_ratio (float): Ratio of data to use for training
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create dataset
    dataset = ChartDataset(real_dir, synthetic_dir)
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader 