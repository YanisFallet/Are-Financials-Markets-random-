import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    """CNN model for financial chart image classification"""
    
    def __init__(self, in_channels=1, num_classes=2):
        """
        Initialize the CNN model
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            num_classes (int): Number of output classes
        """
        super(CNNClassifier, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        # Assuming input size is 224x224
        # After 4 max-pooling layers with stride 2: 224 -> 112 -> 56 -> 28 -> 14
        self.fc_input_size = 256 * 14 * 14
        
        # Classification layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """Forward pass through the network"""
        # Feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class ShallowCNN(nn.Module):
    """A simpler CNN model for financial chart classification"""
    
    def __init__(self, in_channels=1, num_classes=2):
        """
        Initialize a simpler CNN model
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            num_classes (int): Number of output classes
        """
        super(ShallowCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        # For input size 224x224:
        # After conv1(k=5,s=1): 220x220
        # After pool1(k=2,s=2): 110x110
        # After conv2(k=5,s=1): 106x106
        # After pool2(k=2,s=2): 53x53
        # After conv3(k=3,s=1): 51x51
        # After pool3(k=2,s=2): 25x25
        self.fc_input_size = 128 * 25 * 25
        
        # Classification layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """Forward pass through the network"""
        # Feature extraction
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
