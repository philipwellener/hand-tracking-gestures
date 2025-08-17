"""
Dataset loader for hand landmarks.
"""
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2


class HandLandmarkDataset(Dataset):
    """Dataset class for hand landmark detection."""
    
    def __init__(self, data_dir: str, annotations_file: str, transform: Optional[Callable] = None,
                 image_size: Tuple[int, int] = (224, 224), normalize_landmarks: bool = True):
        """
        Initialize the hand landmark dataset.
        
        Args:
            data_dir: Directory containing images
            annotations_file: Path to annotations JSON file
            transform: Data augmentation transforms
            image_size: Target image size (height, width)
            normalize_landmarks: Whether to normalize landmarks to [0, 1]
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.normalize_landmarks = normalize_landmarks
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
    
    def _get_default_transforms(self) -> A.Compose:
        """Get default data augmentation transforms."""
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, landmarks)
        """
        # Get annotation
        annotation = self.annotations[idx]
        image_path = os.path.join(self.data_dir, annotation['image'])
        landmarks = np.array(annotation['landmarks'], dtype=np.float32)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original image dimensions
        original_height, original_width = image.shape[:2]
        
        # Convert landmarks to keypoints format for albumentations
        keypoints = [(x, y) for x, y in landmarks]
        
        # Apply transforms
        transformed = self.transform(image=image, keypoints=keypoints)
        transformed_image = transformed['image']
        transformed_keypoints = transformed['keypoints']
        
        # Convert keypoints back to landmarks array
        landmarks = np.array(transformed_keypoints, dtype=np.float32)
        
        # Normalize landmarks if requested
        if self.normalize_landmarks:
            landmarks[:, 0] = landmarks[:, 0] / self.image_size[1]  # x coordinates
            landmarks[:, 1] = landmarks[:, 1] / self.image_size[0]  # y coordinates
        
        # Convert to tensor
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)
        
        return transformed_image, landmarks_tensor


class SyntheticHandDataset(Dataset):
    """Synthetic hand dataset for testing and prototyping."""
    
    def __init__(self, num_samples: int = 1000, image_size: Tuple[int, int] = (224, 224),
                 num_landmarks: int = 21):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of synthetic samples to generate
            image_size: Image size (height, width)
            num_landmarks: Number of landmarks per hand
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_landmarks = num_landmarks
        
        # Generate synthetic data
        self.images, self.landmarks = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic hand images and landmarks."""
        images = []
        landmarks = []
        
        for _ in range(self.num_samples):
            # Create random background
            img = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
            
            # Generate hand-like landmarks
            hand_landmarks = self._generate_hand_landmarks()
            
            # Draw simple hand representation
            img = self._draw_synthetic_hand(img, hand_landmarks)
            
            images.append(img)
            landmarks.append(hand_landmarks)
        
        return np.array(images), np.array(landmarks)
    
    def _generate_hand_landmarks(self) -> np.ndarray:
        """Generate realistic hand landmark positions."""
        # Start with wrist position
        wrist_x = np.random.uniform(0.3, 0.7) * self.image_size[1]
        wrist_y = np.random.uniform(0.3, 0.7) * self.image_size[0]
        
        landmarks = np.zeros((self.num_landmarks, 2))
        landmarks[0] = [wrist_x, wrist_y]  # Wrist
        
        # Generate finger landmarks
        finger_angles = np.random.uniform(-np.pi/4, np.pi/4, 5)  # 5 fingers
        finger_lengths = np.random.uniform(40, 80, 5)
        
        landmark_idx = 1
        for finger_idx in range(5):
            angle = finger_angles[finger_idx]
            length = finger_lengths[finger_idx]
            
            # Number of joints per finger
            joints_per_finger = 4 if finger_idx == 0 else 4  # Thumb vs other fingers
            
            for joint_idx in range(joints_per_finger):
                # Calculate joint position
                joint_ratio = (joint_idx + 1) / joints_per_finger
                joint_length = length * joint_ratio
                
                x = wrist_x + joint_length * np.cos(angle)
                y = wrist_y + joint_length * np.sin(angle)
                
                # Add some random variation
                x += np.random.normal(0, 5)
                y += np.random.normal(0, 5)
                
                # Ensure landmarks stay within image bounds
                x = np.clip(x, 0, self.image_size[1] - 1)
                y = np.clip(y, 0, self.image_size[0] - 1)
                
                landmarks[landmark_idx] = [x, y]
                landmark_idx += 1
        
        return landmarks.astype(np.float32)
    
    def _draw_synthetic_hand(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw a simple hand representation on the image."""
        # Draw landmarks as circles
        for x, y in landmarks:
            cv2.circle(image, (int(x), int(y)), 3, (255, 255, 255), -1)
        
        # Draw simple connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),    # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = tuple(landmarks[start_idx].astype(int))
                end_point = tuple(landmarks[end_idx].astype(int))
                cv2.line(image, start_point, end_point, (200, 200, 200), 2)
        
        return image
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, landmarks)
        """
        image = self.images[idx]
        landmarks = self.landmarks[idx]
        
        # Convert to tensor and normalize
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)
        
        # Normalize landmarks to [0, 1]
        landmarks_tensor[:, 0] = landmarks_tensor[:, 0] / self.image_size[1]
        landmarks_tensor[:, 1] = landmarks_tensor[:, 1] / self.image_size[0]
        
        return image_tensor, landmarks_tensor


class MediaPipeDataset(Dataset):
    """Dataset that uses MediaPipe for generating hand landmarks."""
    
    def __init__(self, image_dir: str, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize MediaPipe dataset.
        
        Args:
            image_dir: Directory containing hand images
            image_size: Target image size (height, width)
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Initialize MediaPipe
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5
            )
        except ImportError:
            raise ImportError("MediaPipe is required for MediaPipeDataset")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, landmarks) or None if no hand detected
        """
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        
        if image is None:
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None  # No hand detected
        
        # Get first hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract landmark coordinates
        landmarks = []
        for landmark in hand_landmarks.landmark:
            x = landmark.x * image.shape[1]
            y = landmark.y * image.shape[0]
            landmarks.append([x, y])
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Resize image
        image_resized = cv2.resize(image_rgb, (self.image_size[1], self.image_size[0]))
        
        # Scale landmarks to resized image
        scale_x = self.image_size[1] / image.shape[1]
        scale_y = self.image_size[0] / image.shape[0]
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y
        
        # Convert to tensors
        image_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)
        
        # Normalize landmarks to [0, 1]
        landmarks_tensor[:, 0] = landmarks_tensor[:, 0] / self.image_size[1]
        landmarks_tensor[:, 1] = landmarks_tensor[:, 1] / self.image_size[0]
        
        return image_tensor, landmarks_tensor


def create_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True,
                     num_workers: int = 4) -> DataLoader:
    """
    Create a DataLoader for the given dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )


def create_annotation_file(image_dir: str, output_file: str, use_mediapipe: bool = True):
    """
    Create annotation file for a directory of hand images.
    
    Args:
        image_dir: Directory containing hand images
        output_file: Path to output JSON annotation file
        use_mediapipe: Whether to use MediaPipe for automatic annotation
    """
    annotations = []
    
    if use_mediapipe:
        try:
            import mediapipe as mp
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5
            )
            
            for filename in os.listdir(image_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                image_path = os.path.join(image_dir, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    continue
                
                # Process with MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x * image.shape[1]
                        y = landmark.y * image.shape[0]
                        landmarks.append([x, y])
                    
                    annotations.append({
                        'image': filename,
                        'landmarks': landmarks
                    })
                    
                    print(f"Processed {filename}: {len(landmarks)} landmarks")
        
        except ImportError:
            print("MediaPipe not available. Please install it for automatic annotation.")
            return
    
    # Save annotations
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created annotation file: {output_file}")
    print(f"Total annotations: {len(annotations)}")


if __name__ == "__main__":
    # Test synthetic dataset
    print("Testing Synthetic Dataset...")
    synthetic_dataset = SyntheticHandDataset(num_samples=100)
    synthetic_loader = create_dataloader(synthetic_dataset, batch_size=8)
    
    for batch_idx, (images, landmarks) in enumerate(synthetic_loader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Landmarks shape: {landmarks.shape}")
        if batch_idx >= 2:  # Test only a few batches
            break
    
    print("Synthetic dataset test completed!")
