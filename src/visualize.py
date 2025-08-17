"""
Visualization utilities for hand tracking and gesture recognition.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


# Hand landmark connections for drawing skeleton
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17)
]

# Colors for different parts of the hand
LANDMARK_COLORS = {
    'thumb': (255, 0, 0),      # Red
    'index': (0, 255, 0),      # Green
    'middle': (0, 0, 255),     # Blue
    'ring': (255, 255, 0),     # Cyan
    'pinky': (255, 0, 255),    # Magenta
    'palm': (0, 255, 255)      # Yellow
}

# Landmark indices for each finger
FINGER_LANDMARKS = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20],
    'palm': [0]
}


def get_landmark_color(landmark_idx: int) -> Tuple[int, int, int]:
    """Get color for a specific landmark based on finger classification."""
    for finger, indices in FINGER_LANDMARKS.items():
        if landmark_idx in indices:
            return LANDMARK_COLORS[finger]
    return (128, 128, 128)  # Default gray


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, 
                  radius: int = 3, thickness: int = -1) -> np.ndarray:
    """
    Draw hand landmarks on the image.
    
    Args:
        image: Input image
        landmarks: Array of shape (21, 2) with landmark coordinates
        radius: Circle radius for landmarks
        thickness: Circle thickness (-1 for filled)
        
    Returns:
        Image with landmarks drawn
    """
    image_copy = image.copy()
    
    for i, (x, y) in enumerate(landmarks):
        color = get_landmark_color(i)
        cv2.circle(image_copy, (int(x), int(y)), radius, color, thickness)
        
        # Add landmark number for debugging
        if radius > 2:
            cv2.putText(image_copy, str(i), (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return image_copy


def draw_skeleton(image: np.ndarray, landmarks: np.ndarray, 
                 thickness: int = 2) -> np.ndarray:
    """
    Draw hand skeleton connections.
    
    Args:
        image: Input image
        landmarks: Array of shape (21, 2) with landmark coordinates
        thickness: Line thickness
        
    Returns:
        Image with skeleton drawn
    """
    image_copy = image.copy()
    
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = tuple(landmarks[start_idx].astype(int))
            end_point = tuple(landmarks[end_idx].astype(int))
            
            # Use color based on the finger
            color = get_landmark_color(start_idx)
            cv2.line(image_copy, start_point, end_point, color, thickness)
    
    return image_copy


def draw_bounding_box(image: np.ndarray, landmarks: np.ndarray, 
                     padding: int = 20, color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box around hand landmarks.
    
    Args:
        image: Input image
        landmarks: Array of shape (21, 2) with landmark coordinates
        padding: Padding around landmarks
        color: Box color (B, G, R)
        thickness: Box line thickness
        
    Returns:
        Image with bounding box drawn
    """
    image_copy = image.copy()
    
    if len(landmarks) == 0:
        return image_copy
    
    # Calculate bounding box
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    
    x_min = max(0, int(np.min(x_coords)) - padding)
    y_min = max(0, int(np.min(y_coords)) - padding)
    x_max = min(image.shape[1], int(np.max(x_coords)) + padding)
    y_max = min(image.shape[0], int(np.max(y_coords)) + padding)
    
    cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, thickness)
    
    return image_copy


def draw_gesture_label(image: np.ndarray, gesture: str, confidence: float = 0.0,
                      position: Tuple[int, int] = (10, 30),
                      font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
    """
    Draw gesture label on the image.
    
    Args:
        image: Input image
        gesture: Gesture name
        confidence: Gesture confidence (0-1)
        position: Text position (x, y)
        font_scale: Font scale factor
        color: Text color (B, G, R)
        thickness: Text thickness
        
    Returns:
        Image with gesture label drawn
    """
    image_copy = image.copy()
    
    if confidence > 0:
        text = f"{gesture} ({confidence:.2f})"
    else:
        text = gesture
    
    cv2.putText(image_copy, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)
    
    return image_copy


def draw_fps(image: np.ndarray, fps: float, 
            position: Tuple[int, int] = (10, 70),
            font_scale: float = 0.7, color: Tuple[int, int, int] = (255, 255, 255),
            thickness: int = 2) -> np.ndarray:
    """
    Draw FPS counter on the image.
    
    Args:
        image: Input image
        fps: Current FPS value
        position: Text position (x, y)
        font_scale: Font scale factor
        color: Text color (B, G, R)
        thickness: Text thickness
        
    Returns:
        Image with FPS counter drawn
    """
    image_copy = image.copy()
    
    text = f"FPS: {fps:.1f}"
    cv2.putText(image_copy, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)
    
    return image_copy


def create_visualization_overlay(image: np.ndarray, landmarks: np.ndarray,
                               gesture: str, confidence: float, fps: float,
                               show_landmarks: bool = True, show_skeleton: bool = True,
                               show_bbox: bool = True) -> np.ndarray:
    """
    Create complete visualization overlay with landmarks, skeleton, and text.
    
    Args:
        image: Input image
        landmarks: Hand landmarks array of shape (21, 2)
        gesture: Detected gesture name
        confidence: Gesture confidence
        fps: Current FPS
        show_landmarks: Whether to show landmark points
        show_skeleton: Whether to show skeleton connections
        show_bbox: Whether to show bounding box
        
    Returns:
        Image with complete visualization overlay
    """
    result_image = image.copy()
    
    if len(landmarks) > 0:
        # Draw bounding box
        if show_bbox:
            result_image = draw_bounding_box(result_image, landmarks)
        
        # Draw skeleton
        if show_skeleton:
            result_image = draw_skeleton(result_image, landmarks)
        
        # Draw landmarks
        if show_landmarks:
            result_image = draw_landmarks(result_image, landmarks)
    
    # Draw text overlays
    result_image = draw_gesture_label(result_image, gesture, confidence)
    result_image = draw_fps(result_image, fps)
    
    return result_image


def save_frame_with_timestamp(image: np.ndarray, output_dir: str, 
                             prefix: str = "frame") -> str:
    """
    Save frame with timestamp.
    
    Args:
        image: Image to save
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Path to saved image
    """
    import os
    import time
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = int(time.time() * 1000)  # milliseconds
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    cv2.imwrite(filepath, image)
    return filepath


def create_demo_grid(images: List[np.ndarray], titles: List[str] = None,
                    grid_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Create a grid of images for demonstration purposes.
    
    Args:
        images: List of images to combine
        titles: Optional list of titles for each image
        grid_size: Optional grid size (rows, cols). Auto-calculated if None
        
    Returns:
        Combined grid image
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    n_images = len(images)
    
    if grid_size is None:
        # Auto-calculate grid size
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    # Get image dimensions (assume all images are the same size)
    h, w = images[0].shape[:2]
    
    # Create grid
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        # Ensure image has 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Resize if necessary
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        
        # Place image in grid
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        # Add title if provided
        if titles and i < len(titles):
            cv2.putText(grid, titles[i], (col*w + 10, row*h + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return grid


def visualize_swipe_trajectory(frame: np.ndarray, landmark_history: List[np.ndarray], 
                              detected_gesture: Optional[str] = None) -> np.ndarray:
    """
    Draw the hand trajectory for swipe visualization.
    
    Args:
        frame: Input frame
        landmark_history: List of landmark arrays for trajectory
        detected_gesture: Detected swipe gesture if any
        
    Returns:
        Frame with trajectory visualization
    """
    if len(landmark_history) < 2:
        return frame
    
    result_frame = frame.copy()
    
    # Get index fingertip positions (landmark 8)
    positions = [landmarks[8] for landmarks in landmark_history]
    
    # Convert to pixel coordinates
    height, width = frame.shape[:2]
    pixel_positions = []
    
    for pos in positions:
        x = int(pos[0] if pos[0] <= 1.0 else pos[0])  # Handle normalized vs pixel coords
        y = int(pos[1] if pos[1] <= 1.0 else pos[1])
        
        # If coordinates are normalized, convert to pixels
        if pos[0] <= 1.0:
            x = int(pos[0] * width)
            y = int(pos[1] * height)
        
        pixel_positions.append((x, y))
    
    # Draw trajectory line with fading effect
    for i in range(1, len(pixel_positions)):
        # Fade color based on age (newer = more opaque)
        alpha = i / len(pixel_positions)
        color = (0, int(255 * alpha), int(255 * (1 - alpha)))  # Green to red gradient
        thickness = max(1, int(3 * alpha))
        cv2.line(result_frame, pixel_positions[i-1], pixel_positions[i], color, thickness)
    
    # Draw start and end points
    if len(pixel_positions) >= 2:
        cv2.circle(result_frame, pixel_positions[0], 8, (0, 255, 0), -1)  # Green start
        cv2.circle(result_frame, pixel_positions[-1], 8, (0, 0, 255), -1)  # Red end
    
    # Show detected swipe gesture
    if detected_gesture:
        cv2.putText(result_frame, f"SWIPE: {detected_gesture.upper()}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # Add movement statistics
    if len(landmark_history) >= 2:
        start_pos = landmark_history[0][8]
        end_pos = landmark_history[-1][8]
        movement = end_pos - start_pos
        distance = np.linalg.norm(movement)
        velocity = distance / len(landmark_history) if len(landmark_history) > 0 else 0
        
        # Display debug info
        debug_y = height - 120
        cv2.putText(result_frame, f"History: {len(landmark_history)}", 
                   (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result_frame, f"Distance: {distance:.1f}px", 
                   (10, debug_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result_frame, f"Velocity: {velocity:.2f}px/frame", 
                   (10, debug_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result_frame, f"Movement: [{movement[0]:.1f}, {movement[1]:.1f}]", 
                   (10, debug_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return result_frame
