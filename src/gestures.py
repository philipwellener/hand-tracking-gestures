"""
Gesture classification logic from hand landmarks.
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import os


class GestureClassifier:
    """Hand gesture classifier using rule-based and ML approaches."""
    
    def __init__(self, use_ml_classifier: bool = True):
        """
        Initialize gesture classifier.
        
        Args:
            use_ml_classifier: Whether to use ML classifier or rule-based only
        """
        self.use_ml_classifier = use_ml_classifier
        self.ml_model = None
        self.scaler = None
        self.gesture_history = []
        self.history_size = 5  # For temporal smoothing
        
        # Initialize ML classifier if requested
        if use_ml_classifier:
            self._initialize_ml_classifier()
    
    def _initialize_ml_classifier(self):
        """Initialize the machine learning classifier."""
        self.ml_model = SVC(probability=True, kernel='rbf', random_state=42)
        self.scaler = StandardScaler()
    
    def extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from hand landmarks.
        
        Args:
            landmarks: Array of shape (21, 2) with landmark coordinates
            
        Returns:
            Feature vector
        """
        if landmarks.shape != (21, 2):
            raise ValueError("Expected landmarks shape (21, 2)")
        
        features = []
        
        # Normalize landmarks relative to wrist (landmark 0)
        wrist = landmarks[0]
        normalized_landmarks = landmarks - wrist
        
        # Flatten normalized coordinates
        features.extend(normalized_landmarks.flatten())
        
        # Distance features
        distances = self._calculate_distances(landmarks)
        features.extend(distances)
        
        # Angle features
        angles = self._calculate_angles(landmarks)
        features.extend(angles)
        
        # Finger extension features
        extensions = self._calculate_finger_extensions(landmarks)
        features.extend(extensions)
        
        return np.array(features)
    
    def _calculate_distances(self, landmarks: np.ndarray) -> List[float]:
        """Calculate distance-based features."""
        distances = []
        
        # Distances from wrist to fingertips
        wrist = landmarks[0]
        fingertips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
        
        for tip_idx in fingertips:
            dist = np.linalg.norm(landmarks[tip_idx] - wrist)
            distances.append(dist)
        
        # Distances between fingertips
        for i, tip1 in enumerate(fingertips):
            for tip2 in fingertips[i+1:]:
                dist = np.linalg.norm(landmarks[tip1] - landmarks[tip2])
                distances.append(dist)
        
        # Palm width and height
        palm_points = [0, 5, 9, 13, 17]  # Wrist and finger bases
        palm_coords = landmarks[palm_points]
        palm_width = np.max(palm_coords[:, 0]) - np.min(palm_coords[:, 0])
        palm_height = np.max(palm_coords[:, 1]) - np.min(palm_coords[:, 1])
        distances.extend([palm_width, palm_height])
        
        return distances
    
    def _calculate_angles(self, landmarks: np.ndarray) -> List[float]:
        """Calculate angle-based features."""
        angles = []
        
        # Finger joint angles
        finger_joints = [
            [1, 2, 3],    # Thumb
            [5, 6, 7],    # Index
            [9, 10, 11],  # Middle
            [13, 14, 15], # Ring
            [17, 18, 19]  # Pinky
        ]
        
        for joints in finger_joints:
            for i in range(len(joints) - 2):
                p1, p2, p3 = landmarks[joints[i]], landmarks[joints[i+1]], landmarks[joints[i+2]]
                angle = self._calculate_angle(p1, p2, p3)
                angles.append(angle)
        
        return angles
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def _calculate_finger_extensions(self, landmarks: np.ndarray) -> List[float]:
        """
        Calculate finger extension using angle-based approach.
        """
        extensions = []
        
        # MediaPipe hand landmark structure:
        fingers = [
            [1, 2, 3, 4],      # Thumb
            [5, 6, 7, 8],      # Index
            [9, 10, 11, 12],   # Middle
            [13, 14, 15, 16],  # Ring
            [17, 18, 19, 20]   # Pinky
        ]
        
        for finger_idx, finger_landmarks in enumerate(fingers):
            joints = [landmarks[i] for i in finger_landmarks]
            
            # Calculate angles at each joint
            angles = []
            for i in range(1, 3):  # Middle two joints (skip base and tip)
                # Get three consecutive points
                p1, p2, p3 = joints[i-1], joints[i], joints[i+1]
                
                # Calculate vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Calculate angle
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 0 and v2_norm > 0:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
                else:
                    angles.append(np.pi/2)  # Default to bent if vectors are zero
            
            # Average angle - straight finger has angles close to π (180°)
            if len(angles) > 0:
                avg_angle = np.mean(angles)
            else:
                avg_angle = np.pi/2
            
            # Convert to extension score
            # Straight finger: angle ≈ π → extension = 1
            # Bent finger: angle ≈ π/2 → extension = 0
            extension = (avg_angle - np.pi/2) / (np.pi/2)
            extension = np.clip(extension, 0.0, 1.0)
            
            extensions.append(extension)
        
        return extensions
    
    def classify_gesture_rules(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Classify gesture using rule-based approach.
        
        Args:
            landmarks: Array of shape (21, 2) with landmark coordinates
            
        Returns:
            Tuple of (gesture_name, confidence)
        """
        features = self.extract_features(landmarks)
        
        # Extract specific features for rule-based classification
        fingertip_distances = features[42:47]  # Distances from wrist to fingertips
        finger_extensions = features[-5:]      # Finger extension ratios
        
        # Rule-based gesture classification with improved thresholds for simple approach
        
        # 6. Pinch - thumb and index close together (check BEFORE open palm)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_distance = np.linalg.norm(thumb_tip - index_tip)
        
        # Calculate relative pinch distance based on palm size
        palm_width = np.linalg.norm(landmarks[17] - landmarks[5])
        relative_pinch = pinch_distance / (palm_width + 1e-8)
        
        # Pinch detection - more lenient thresholds
        if pinch_distance < 60 and relative_pinch < 0.4:  # Increased from 120 and 0.3
            return "pinch", 0.85
        
        # 7. OK Sign - thumb and index forming circle, others extended  
        if (pinch_distance < 120 and  # Increased from 200 (more lenient circle)
            relative_pinch < 0.7 and  # Increased from 0.5 (more lenient)
            np.all(finger_extensions[2:] > 0.5)):  # Lowered from 0.6 (easier for other fingers)
            return "ok", 0.8
        
        # 1. Open Palm - all fingers extended (moved after pinch/ok checks)
        if np.all(finger_extensions > 0.5):  # Realistic threshold based on data
            return "open_palm", 0.95
        
        # 2. Fist - all fingers bent (check this BEFORE thumbs_up)
        if (np.all(finger_extensions[1:] < 0.2) and  # All non-thumb fingers very bent
            finger_extensions[0] < 0.6):  # Thumb somewhat bent (more lenient)
            return "fist", 0.9
        
        # 3. Thumbs Up - only thumb extended
        if (finger_extensions[0] > 0.5 and  # Thumb extended (was 0.4)
            np.all(finger_extensions[1:] < 0.3)):  # Other fingers bent
            return "thumbs_up", 0.95
        
        # 4. Peace Sign - index and middle extended, others bent
        if (finger_extensions[1] > 0.8 and  # Index extended (was 0.4, now higher)
            finger_extensions[2] > 0.8 and  # Middle extended (was 0.4, now higher)
            finger_extensions[0] < 0.8 and  # Thumb bent (was 0.35, more lenient)
            finger_extensions[3] < 0.2 and  # Ring bent (was 0.35, stricter)
            finger_extensions[4] < 0.2):    # Pinky bent (was 0.35, stricter)
            return "peace", 0.9
        
        # 5. Pointing - only index extended
        if (finger_extensions[1] > 0.8 and  # Index extended (was 0.4, now higher)
            finger_extensions[0] < 0.8 and  # Thumb bent (was 0.35, more lenient)
            finger_extensions[2] < 0.2 and  # Middle bent (was 0.35, stricter)
            finger_extensions[3] < 0.2 and  # Ring bent (was 0.35, stricter)
            finger_extensions[4] < 0.2):    # Pinky bent (was 0.35, stricter)
            return "pointing", 0.85
        
        # Default: Unknown gesture
        return "unknown", 0.1
    
    def classify_gesture_ml(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Classify gesture using machine learning approach.
        
        Args:
            landmarks: Array of shape (21, 2) with landmark coordinates
            
        Returns:
            Tuple of (gesture_name, confidence)
        """
        if self.ml_model is None or self.scaler is None:
            return "unknown", 0.0
        
        try:
            features = self.extract_features(landmarks)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            predicted_class = self.ml_model.classes_[np.argmax(probabilities)]
            confidence = np.max(probabilities)
            
            return predicted_class, confidence
        
        except Exception as e:
            print(f"ML classification error: {e}")
            return "unknown", 0.0
    
    def classify_gesture(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Classify gesture using the configured approach.
        
        Args:
            landmarks: Array of shape (21, 2) with landmark coordinates
            
        Returns:
            Tuple of (gesture_name, confidence)
        """
        if self.use_ml_classifier and self.ml_model is not None:
            gesture, confidence = self.classify_gesture_ml(landmarks)
            
            # Fallback to rule-based if ML confidence is low
            if confidence < 0.5:
                gesture, confidence = self.classify_gesture_rules(landmarks)
        else:
            gesture, confidence = self.classify_gesture_rules(landmarks)
        
        # Apply temporal smoothing
        gesture = self._apply_temporal_smoothing(gesture)
        
        return gesture, confidence
    
    def _apply_temporal_smoothing(self, current_gesture: str) -> str:
        """Apply temporal smoothing to reduce gesture flickering."""
        self.gesture_history.append(current_gesture)
        
        # Keep only recent history
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # Return most common gesture in recent history with improved logic
        if len(self.gesture_history) >= 3:
            gesture_counts = {}
            for gesture in self.gesture_history:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            
            # Prioritize non-unknown gestures
            non_unknown = {k: v for k, v in gesture_counts.items() if k != "unknown"}
            if non_unknown:
                most_common = max(non_unknown.items(), key=lambda x: x[1])
                # Only return if it appears at least 40% of the time
                if most_common[1] / len(self.gesture_history) >= 0.4:
                    return most_common[0]
            
            # Fallback to original logic
            most_common = max(gesture_counts.items(), key=lambda x: x[1])
            return most_common[0]
        
        return current_gesture
    
    def train_ml_classifier(self, X: np.ndarray, y: np.ndarray):
        """
        Train the ML classifier.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
        """
        if not self.use_ml_classifier:
            print("ML classifier is disabled")
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.ml_model.fit(X_scaled, y)
        
        print(f"Trained ML classifier on {len(X)} samples")
        print(f"Classes: {self.ml_model.classes_}")
    
    def save_model(self, model_path: str):
        """Save trained ML model and scaler."""
        if self.ml_model is None or self.scaler is None:
            print("No trained model to save")
            return
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.ml_model,
            'scaler': self.scaler,
            'classes': self.ml_model.classes_
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained ML model and scaler."""
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return
        
        model_data = joblib.load(model_path)
        self.ml_model = model_data['model']
        self.scaler = model_data['scaler']
        
        print(f"Model loaded from {model_path}")
        print(f"Classes: {self.ml_model.classes_}")


def detect_swipe_gesture(landmark_history: List[np.ndarray], 
                        min_movement: float = 80.0,
                        frame_width: int = 1000,
                        frame_height: int = 1000) -> Optional[str]:
    """
    Detect swipe gestures from landmark history with improved filtering.
    
    Args:
        landmark_history: List of recent landmark arrays (normalized 0-1 coordinates)
        min_movement: Minimum movement threshold for swipe detection (in pixels)
        frame_width: Width of frame for pixel conversion
        frame_height: Height of frame for pixel conversion
        
    Returns:
        Swipe direction ('swipe_left', 'swipe_right', 'swipe_up', 'swipe_down') or None
    """
    if len(landmark_history) < 15:  # Need more history for reliable swipe detection
        return None
    
    # Use index fingertip for swipe detection
    start_pos = landmark_history[0][8]  # Index fingertip (normalized coords)
    end_pos = landmark_history[-1][8]
    
    # Convert to pixel coordinates for distance calculation
    start_pixels = np.array([start_pos[0] * frame_width, start_pos[1] * frame_height])
    end_pixels = np.array([end_pos[0] * frame_width, end_pos[1] * frame_height])
    
    # Calculate movement vector in pixels
    movement_pixels = end_pixels - start_pixels
    distance_pixels = np.linalg.norm(movement_pixels)
    
    if distance_pixels < min_movement:
        return None
    
    # Check for consistent movement direction (reduce false positives)
    mid_point = landmark_history[len(landmark_history)//2][8]
    mid_pixels = np.array([mid_point[0] * frame_width, mid_point[1] * frame_height])
    
    movement_1_pixels = mid_pixels - start_pixels
    movement_2_pixels = end_pixels - mid_pixels
    
    # Both movements should be in similar direction
    dot_product = np.dot(movement_1_pixels, movement_2_pixels)
    if dot_product < 0:  # Movements in opposite directions
        return None
    
    # Calculate velocity - swipes should be relatively fast
    velocity_pixels = distance_pixels / len(landmark_history)
    if velocity_pixels < 10.0:  # Increased threshold for more reliable swipe detection
        return None
    
    # Determine primary direction using normalized movement for direction consistency
    movement_normalized = end_pos - start_pos
    dx, dy = movement_normalized
    
    if abs(dx) > abs(dy):
        # Horizontal movement
        if dx > 0:
            return "swipe_right"
        else:
            return "swipe_left"
    else:
        # Vertical movement
        if dy > 0:
            return "swipe_down"
        else:
            return "swipe_up"
