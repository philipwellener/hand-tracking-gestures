"""
Utility functions for hand tracking and gesture recognition project.
"""
import os
import time
import numpy as np
import cv2
import torch
from typing import Tuple, List, Optional, Union


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str = 'cpu') -> dict:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return checkpoint


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, checkpoint_dir: str, 
                   filename: Optional[str] = None) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        checkpoint_dir: Directory to save checkpoint
        filename: Optional filename, defaults to epoch-based naming
        
    Returns:
        Path to saved checkpoint
    """
    ensure_dir(checkpoint_dir)
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }, checkpoint_path)
    
    return checkpoint_path


def normalize_landmarks(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Normalize landmark coordinates to [0, 1] range.
    
    Args:
        landmarks: Array of shape (21, 2) with (x, y) coordinates
        image_shape: (height, width) of the image
        
    Returns:
        Normalized landmarks
    """
    height, width = image_shape
    normalized = landmarks.copy()
    normalized[:, 0] = landmarks[:, 0] / width  # x coordinates
    normalized[:, 1] = landmarks[:, 1] / height  # y coordinates
    return normalized


def denormalize_landmarks(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Denormalize landmark coordinates from [0, 1] range to pixel coordinates.
    
    Args:
        landmarks: Normalized array of shape (21, 2)
        image_shape: (height, width) of the image
        
    Returns:
        Denormalized landmarks in pixel coordinates
    """
    height, width = image_shape
    denormalized = landmarks.copy()
    denormalized[:, 0] = landmarks[:, 0] * width  # x coordinates
    denormalized[:, 1] = landmarks[:, 1] * height  # y coordinates
    return denormalized.astype(int)


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(point1 - point2)


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle between three points (p1-p2-p3).
    
    Args:
        p1, p2, p3: Points as numpy arrays of shape (2,)
        
    Returns:
        Angle in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
    
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


class FPSCounter:
    """Simple FPS counter for real-time applications."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []
    
    def update(self) -> float:
        """Update FPS counter and return current FPS."""
        current_time = time.time()
        self.timestamps.append(current_time)
        
        # Keep only recent timestamps
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        fps = (len(self.timestamps) - 1) / time_diff if time_diff > 0 else 0.0
        
        return fps


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image: Input image as numpy array
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed image
    """
    # Resize image
    resized = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    return normalized


def postprocess_landmarks(predictions: torch.Tensor, original_size: Tuple[int, int], 
                         model_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Post-process model predictions to get landmark coordinates.
    
    Args:
        predictions: Model output tensor of shape (21, 2) or (batch, 21, 2)
        original_size: Original image size (height, width)
        model_size: Model input size (height, width)
        
    Returns:
        Landmark coordinates in original image coordinates
    """
    if predictions.dim() == 3:
        predictions = predictions.squeeze(0)  # Remove batch dimension if present
    
    landmarks = predictions.detach().cpu().numpy()
    
    # Scale from model coordinates to original image coordinates
    scale_x = original_size[1] / model_size[1]
    scale_y = original_size[0] / model_size[0]
    
    landmarks[:, 0] *= scale_x
    landmarks[:, 1] *= scale_y
    
    return landmarks.astype(int)


def validate_landmarks(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> bool:
    """
    Validate that landmarks are within image boundaries.
    
    Args:
        landmarks: Array of shape (21, 2)
        image_shape: (height, width) of the image
        
    Returns:
        True if all landmarks are valid
    """
    if landmarks.shape != (21, 2):
        return False
    
    height, width = image_shape
    
    # Check if all landmarks are within image boundaries
    valid_x = np.all((landmarks[:, 0] >= 0) & (landmarks[:, 0] < width))
    valid_y = np.all((landmarks[:, 1] >= 0) & (landmarks[:, 1] < height))
    
    return valid_x and valid_y


def smooth_landmarks(landmarks: np.ndarray, previous_landmarks: Optional[np.ndarray], 
                    alpha: float = 0.7) -> np.ndarray:
    """
    Apply temporal smoothing to landmarks using exponential moving average.
    
    Args:
        landmarks: Current landmarks array of shape (21, 2)
        previous_landmarks: Previous landmarks array of shape (21, 2) or None
        alpha: Smoothing factor (0 = no smoothing, 1 = no history)
        
    Returns:
        Smoothed landmarks
    """
    if previous_landmarks is None:
        return landmarks
    
    return alpha * landmarks + (1 - alpha) * previous_landmarks
