"""
PyTorch model for hand landmark detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HandLandmarkCNN(nn.Module):
    """CNN model for hand landmark detection."""
    
    def __init__(self, num_landmarks: int = 21, input_channels: int = 3):
        """
        Initialize the hand landmark detection model.
        
        Args:
            num_landmarks: Number of landmarks to predict (default: 21)
            input_channels: Number of input channels (default: 3 for RGB)
        """
        super(HandLandmarkCNN, self).__init__()
        
        self.num_landmarks = num_landmarks
        self.input_channels = input_channels
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_landmarks * 2)  # x, y coordinates for each landmark
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Landmark coordinates of shape (batch_size, num_landmarks, 2)
        """
        # Convolutional layers with batch normalization and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        # Reshape to (batch_size, num_landmarks, 2)
        x = x.view(-1, self.num_landmarks, 2)
        
        return x


class ResNetBackbone(nn.Module):
    """ResNet-based backbone for feature extraction."""
    
    def __init__(self, input_channels: int = 3):
        super(ResNetBackbone, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """Create a ResNet layer with multiple blocks."""
        layers = []
        
        # First block may have stride > 1
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        
        # Remaining blocks have stride = 1
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


class ResNetBlock(nn.Module):
    """Basic ResNet block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class HandLandmarkResNet(nn.Module):
    """ResNet-based model for hand landmark detection."""
    
    def __init__(self, num_landmarks: int = 21, input_channels: int = 3):
        """
        Initialize the ResNet-based hand landmark detection model.
        
        Args:
            num_landmarks: Number of landmarks to predict (default: 21)
            input_channels: Number of input channels (default: 3 for RGB)
        """
        super(HandLandmarkResNet, self).__init__()
        
        self.num_landmarks = num_landmarks
        self.backbone = ResNetBackbone(input_channels)
        
        # Regression head for landmark coordinates
        self.fc = nn.Linear(512, num_landmarks * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Landmark coordinates of shape (batch_size, num_landmarks, 2)
        """
        features = self.backbone(x)
        landmarks = self.fc(features)
        landmarks = landmarks.view(-1, self.num_landmarks, 2)
        
        return landmarks


class LightweightHandNet(nn.Module):
    """Lightweight model for real-time hand landmark detection."""
    
    def __init__(self, num_landmarks: int = 21, input_channels: int = 3):
        """
        Initialize the lightweight hand landmark detection model.
        
        Args:
            num_landmarks: Number of landmarks to predict (default: 21)
            input_channels: Number of input channels (default: 3 for RGB)
        """
        super(LightweightHandNet, self).__init__()
        
        self.num_landmarks = num_landmarks
        
        # Depthwise separable convolutions for efficiency
        self.conv1 = self._depthwise_separable_conv(input_channels, 32, stride=2)
        self.conv2 = self._depthwise_separable_conv(32, 64, stride=2)
        self.conv3 = self._depthwise_separable_conv(64, 128, stride=2)
        self.conv4 = self._depthwise_separable_conv(128, 256, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_landmarks * 2)
        
        self.dropout = nn.Dropout(0.3)
        
    def _depthwise_separable_conv(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create a depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Landmark coordinates of shape (batch_size, num_landmarks, 2)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        x = x.view(-1, self.num_landmarks, 2)
        
        return x


def create_model(model_type: str = "lightweight", num_landmarks: int = 21, 
                input_channels: int = 3) -> nn.Module:
    """
    Create a hand landmark detection model.
    
    Args:
        model_type: Type of model ("cnn", "resnet", "lightweight")
        num_landmarks: Number of landmarks to predict
        input_channels: Number of input channels
        
    Returns:
        PyTorch model
    """
    if model_type == "cnn":
        return HandLandmarkCNN(num_landmarks, input_channels)
    elif model_type == "resnet":
        return HandLandmarkResNet(num_landmarks, input_channels)
    elif model_type == "lightweight":
        return LightweightHandNet(num_landmarks, input_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class HandLandmarkLoss(nn.Module):
    """Custom loss function for hand landmark detection."""
    
    def __init__(self, use_wing_loss: bool = True, wing_omega: float = 10.0, wing_epsilon: float = 2.0):
        """
        Initialize the loss function.
        
        Args:
            use_wing_loss: Whether to use Wing loss instead of MSE
            wing_omega: Wing loss omega parameter
            wing_epsilon: Wing loss epsilon parameter
        """
        super(HandLandmarkLoss, self).__init__()
        self.use_wing_loss = use_wing_loss
        self.omega = wing_omega
        self.epsilon = wing_epsilon
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss between predictions and targets.
        
        Args:
            predictions: Predicted landmarks of shape (batch_size, num_landmarks, 2)
            targets: Target landmarks of shape (batch_size, num_landmarks, 2)
            
        Returns:
            Loss value
        """
        if self.use_wing_loss:
            return self._wing_loss(predictions, targets)
        else:
            return F.mse_loss(predictions, targets)
    
    def _wing_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Wing loss for better handling of small errors.
        """
        diff = torch.abs(predictions - targets)
        
        # Wing loss formula
        C = self.omega - self.omega * torch.log(1 + self.omega / self.epsilon)
        
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - C
        )
        
        return loss.mean()


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)):
    """
    Print model summary including parameter count and output shapes.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    x = torch.randn(input_size).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Print summary
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {list(x.shape)}")
    print(f"Output shape: {list(output.shape)}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    print(f"Model size: {size_mb:.2f} MB")


if __name__ == "__main__":
    # Test the models
    models = {
        "CNN": create_model("cnn"),
        "ResNet": create_model("resnet"),
        "Lightweight": create_model("lightweight")
    }
    
    for name, model in models.items():
        print(f"\n{name} Model:")
        model_summary(model)
