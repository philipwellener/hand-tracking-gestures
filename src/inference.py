"""
Real-time hand tracking and gesture recognition inference script.
"""
import cv2
import numpy as np
import torch
import time
import argparse
import os
from typing import Optional, Tuple, List

from model import create_model
from gestures import GestureClassifier, detect_swipe_gesture
from visualize import create_visualization_overlay, save_frame_with_timestamp, visualize_swipe_trajectory
from utils import FPSCounter, preprocess_image, postprocess_landmarks, smooth_landmarks


class HandTrackingInference:
    """Real-time hand tracking and gesture recognition."""
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "lightweight",
                 device: str = "auto", use_mediapipe: bool = False, debug_mode: bool = False,
                 show_trajectory: bool = False, confidence_threshold: float = 0.5):
        """
        Initialize hand tracking inference.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model architecture
            confidence_threshold: Minimum confidence for detections
            use_mediapipe: Whether to use MediaPipe as fallback
            debug_mode: Enable debug information display
            show_trajectory: Show swipe trajectory visualization
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        self.debug_mode = debug_mode
        self.show_trajectory = show_trajectory
        
        # Initialize PyTorch model
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Initialize MediaPipe as fallback
        self.use_mediapipe = use_mediapipe
        self.mp_hands = None
        if use_mediapipe:
            print("Setting up MediaPipe... (this may take a moment)")
            self._initialize_mediapipe()
        else:
            print("MediaPipe disabled (use --enable-mediapipe to enable)")
            print("Note: Will use rule-based gesture detection only")
        
        # Initialize gesture classifier
        self.gesture_classifier = GestureClassifier(use_ml_classifier=False)
        
        # Initialize utilities
        self.fps_counter = FPSCounter()
        self.previous_landmarks = None
        self.landmark_history = []
        self.max_history = 30  # For swipe detection
        
        # Swipe detection cooldown
        self.last_swipe_time = 0
        self.swipe_cooldown = 1.0  # 1 second cooldown between swipes
        
        print(f"Hand tracking inference initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model: {'Custom' if self.model else 'None'}")
        print(f"  MediaPipe: {'Enabled' if self.mp_hands else 'Disabled'}")
    
    def _calculate_hand_stability(self) -> float:
        """Calculate hand movement stability from recent landmark history."""
        if len(self.landmark_history) < 3:
            return 0.0
        
        # Calculate movement between recent frames
        movements = []
        for i in range(len(self.landmark_history) - 2, len(self.landmark_history)):
            if i > 0:
                # Use wrist position (landmark 0) as reference
                prev_wrist = self.landmark_history[i-1][0]
                curr_wrist = self.landmark_history[i][0]
                movement = np.linalg.norm(curr_wrist - prev_wrist)
                movements.append(movement)
        
        return np.mean(movements) if movements else 0.0
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe hands solution."""
        try:
            print("Initializing MediaPipe...")
            import mediapipe as mp
            mp_hands = mp.solutions.hands
            self.mp_hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✓ MediaPipe initialized successfully")
        except Exception as e:
            print(f"✗ MediaPipe initialization failed: {e}")
            print("Continuing without MediaPipe - rule-based detection only")
            self.use_mediapipe = False
            self.mp_hands = None
    
    def load_model(self, model_path: str):
        """Load trained PyTorch model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model with correct architecture
            if 'config' in checkpoint:
                config = checkpoint['config']
                self.model = create_model(
                    model_type=config.get('model_type', self.model_type),
                    num_landmarks=config.get('num_landmarks', 21),
                    input_channels=config.get('input_channels', 3)
                )
            else:
                self.model = create_model(model_type=self.model_type)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded from: {model_path}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
    
    def detect_landmarks_pytorch(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect hand landmarks using PyTorch model."""
        if self.model is None:
            return None
        
        try:
            # Preprocess image
            processed_image = preprocess_image(image, target_size=(224, 224))
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.tensor(processed_image, dtype=torch.float32)
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            input_tensor = input_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                predictions = self.model(input_tensor)
            
            # Post-process predictions
            landmarks = postprocess_landmarks(
                predictions, 
                original_size=image.shape[:2],
                model_size=(224, 224)
            )
            
            return landmarks
            
        except Exception as e:
            print(f"PyTorch inference error: {e}")
            return None
    
    def detect_landmarks_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect hand landmarks using MediaPipe."""
        if self.mp_hands is None:
            return None
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.mp_hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Convert to numpy array
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = landmark.x * image.shape[1]
                    y = landmark.y * image.shape[0]
                    landmarks.append([x, y])
                
                return np.array(landmarks, dtype=np.float32)
            
            return None
            
        except Exception as e:
            print(f"MediaPipe inference error: {e}")
            return None
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect hand landmarks using available methods."""
        landmarks = None
        
        # Try PyTorch model first
        if self.model is not None:
            landmarks = self.detect_landmarks_pytorch(image)
            if landmarks is not None and self.debug_mode:
                print("Using PyTorch landmarks")
        
        # Fallback to MediaPipe
        if landmarks is None and self.use_mediapipe and self.mp_hands is not None:
            landmarks = self.detect_landmarks_mediapipe(image)
            if landmarks is not None and self.debug_mode:
                print("Using MediaPipe landmarks")
        
        return landmarks
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, float, float]:
        """
        Process a single frame for hand tracking and gesture recognition.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, gesture, confidence, fps)
        """
        # Update FPS counter
        fps = self.fps_counter.update()
        
        # Detect landmarks
        landmarks = self.detect_landmarks(frame)
        
        gesture = "no_hand"
        confidence = 0.0
        detected_swipe = None
        current_time = time.time()
        
        if landmarks is not None:
            # Apply temporal smoothing
            landmarks = smooth_landmarks(landmarks, self.previous_landmarks, alpha=0.7)
            self.previous_landmarks = landmarks.copy()
            
            # Update landmark history for swipe detection
            self.landmark_history.append(landmarks)
            if len(self.landmark_history) > self.max_history:
                self.landmark_history.pop(0)
            
            # Classify static gesture
            gesture, confidence = self.gesture_classifier.classify_gesture(landmarks)
            
            # Check for swipe gestures with improved logic
            if len(self.landmark_history) >= 15:
                # Get frame dimensions for swipe detection
                frame_height, frame_width = frame.shape[:2]
                detected_swipe = detect_swipe_gesture(
                    self.landmark_history, 
                    min_movement=100.0,  # Increased threshold for more reliable detection
                    frame_width=frame_width,
                    frame_height=frame_height
                )
                
                # Apply cooldown to reduce noise
                if detected_swipe and current_time - self.last_swipe_time > self.swipe_cooldown:
                    self.last_swipe_time = current_time
                    gesture = detected_swipe
                    confidence = 0.7  # Lower confidence for swipes
                else:
                    detected_swipe = None
            
            # Additional filtering: prefer static gestures when hand is relatively stable
            if len(self.landmark_history) >= 5 and not detected_swipe:
                recent_movement = self._calculate_hand_stability()
                if recent_movement < 20.0 and confidence > 0.5:  # Hand is stable
                    # Keep static gesture, ignore potential swipes
                    pass
        
        # Create visualization
        if landmarks is not None:
            if self.show_trajectory and len(self.landmark_history) > 1:
                # Use trajectory visualization
                processed_frame = visualize_swipe_trajectory(
                    frame, self.landmark_history, detected_swipe)
            else:
                # Use standard visualization
                processed_frame = create_visualization_overlay(
                    frame, landmarks, gesture, confidence, fps,
                    show_landmarks=True, show_skeleton=True, show_bbox=True
                )
        else:
            processed_frame = frame.copy()
            # Add FPS and status text
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, gesture, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        return processed_frame, gesture, confidence, fps
    
    def run_webcam_demo(self, camera_id: int = 0, output_dir: Optional[str] = None,
                       save_frames: bool = False, show_display: bool = True):
        """
        Run real-time webcam demo.
        
        Args:
            camera_id: Camera device ID
            output_dir: Directory to save demo frames/videos
            save_frames: Whether to save sample frames
            show_display: Whether to show live display
        """
        # Initialize camera
        print(f"Attempting to open camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        # Wait a moment for camera to initialize
        time.sleep(1)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            print("Troubleshooting tips:")
            print("1. Check if camera is being used by another app")
            print("2. Try a different camera ID (--camera 1)")
            print("3. Check camera permissions in macOS System Preferences")
            return
        
        print("Camera opened successfully!")
        
        # Set camera properties with error handling
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid lag
        except Exception as e:
            print(f"Warning: Could not set camera properties: {e}")
        
        # Test if we can read a frame with timeout
        print("Testing camera frame capture...")
        for attempt in range(5):  # Try 5 times
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"✓ Camera test successful! Frame shape: {test_frame.shape}")
                break
            else:
                print(f"Attempt {attempt + 1}/5: Could not capture frame")
                time.sleep(0.5)
        else:
            print("Warning: Cannot read frames from camera")
            print("Running in test mode - press 'q' to quit")
            # Create a dummy frame for testing
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "Camera not available", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(dummy_frame, "Press 'q' to quit", (50, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show dummy window and wait for input
            if show_display:
                cv2.namedWindow('Hand Tracking Demo', cv2.WINDOW_NORMAL)
                cv2.imshow('Hand Tracking Demo', dummy_frame)
            
            print("Dummy window created. Press 'q' to quit.")
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("User quit from test mode")
                    break
            
            cap.release()
            if show_display:
                cv2.destroyAllWindows()
            return
        
        print("Starting webcam demo...")
        if self.debug_mode or self.show_trajectory:
            print("=== DEBUG MODE ===")
            print("Instructions:")
            print("- Make clear gestures and swipes")
            print("- Green dot = trajectory start, Red dot = trajectory end")
            print("- Press 'q' to quit, 's' to save frame, 'r' to record video")
            print("- Press 'd' to toggle debug info, 't' to toggle trajectory")
            print("- Press 'c' to clear landmark history")
            print("- IMPORTANT: Click on the camera window to make it active for key detection!")
        else:
            print("Press 'q' to quit, 's' to save frame, 'r' to record video")
            print("IMPORTANT: Click on the camera window to make it active for key detection!")
        
        # Create and setup OpenCV window
        if show_display:
            cv2.namedWindow('Hand Tracking Demo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Hand Tracking Demo', 640, 480)
            # Force window to front on macOS
            cv2.setWindowProperty('Hand Tracking Demo', cv2.WND_PROP_TOPMOST, 1)
            cv2.setWindowProperty('Hand Tracking Demo', cv2.WND_PROP_TOPMOST, 0)
            print("Camera window created - window should be in front for key detection")
        
        # Show a dummy frame first to establish the window
        if show_display:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "Starting camera...", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Hand Tracking Demo', dummy_frame)
            cv2.waitKey(100)  # Give time for window to appear
        
        # Video recording setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = None
        recording = False
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Could not read frame from camera")
                    # Check for user input even when camera fails
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        print("User pressed 'q' to quit")
                        break
                    continue
                
                # Process frame
                processed_frame, gesture, confidence, fps = self.process_frame(frame)
                
                # Display frame
                if show_display:
                    cv2.imshow('Hand Tracking Demo', processed_frame)
                    # Ensure window is active and responsive
                    cv2.setWindowProperty('Hand Tracking Demo', cv2.WND_PROP_TOPMOST, 1)
                    cv2.setWindowProperty('Hand Tracking Demo', cv2.WND_PROP_TOPMOST, 0)
                
                # Handle video recording
                if recording and video_writer is not None:
                    video_writer.write(processed_frame)
                
                # Handle keyboard input - CRITICAL: Always check for input
                key = cv2.waitKey(50) & 0xFF
                
                if key == ord('q') or key == 113:  # 'q' or direct ASCII
                    print("Quit command detected!")
                    break
                elif key == ord('s') and save_frames and output_dir:
                    # Save current frame
                    filename = save_frame_with_timestamp(processed_frame, output_dir, "demo_frame")
                    print(f"Frame saved: {filename}")
                elif key == ord('r'):
                    # Toggle recording
                    if not recording:
                        if output_dir:
                            os.makedirs(output_dir, exist_ok=True)
                            video_path = os.path.join(output_dir, f"demo_video_{int(time.time())}.mp4")
                            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, 
                                                         (processed_frame.shape[1], processed_frame.shape[0]))
                            recording = True
                            print(f"Recording started: {video_path}")
                    else:
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                        recording = False
                        print("Recording stopped")
                elif key == ord('d'):
                    # Toggle debug mode
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('t'):
                    # Toggle trajectory visualization
                    self.show_trajectory = not self.show_trajectory
                    print(f"Trajectory visualization: {'ON' if self.show_trajectory else 'OFF'}")
                elif key == ord('c'):
                    # Clear landmark history
                    self.landmark_history.clear()
                    print("Landmark history cleared")
                elif key == ord('p') and self.debug_mode:
                    # Print current parameters
                    print("\n=== CURRENT PARAMETERS ===")
                    print(f"Min movement threshold: 100.0px")
                    print(f"Min velocity threshold: 10.0px/frame")
                    print(f"Min history length: 15 frames")
                    print(f"Current history: {len(self.landmark_history)} frames")
                    if len(self.landmark_history) >= 2:
                        start_pos = self.landmark_history[0][8]
                        end_pos = self.landmark_history[-1][8]
                        movement = end_pos - start_pos
                        distance = np.linalg.norm(movement)
                        velocity = distance / len(self.landmark_history)
                        print(f"Current distance: {distance:.1f}px")
                        print(f"Current velocity: {velocity:.2f}px/frame")
                    print("========================\n")
                
                frame_count += 1
                
                # Auto-save sample frames periodically
                if save_frames and output_dir and frame_count % 300 == 0:  # Every 10 seconds at 30 FPS
                    save_frame_with_timestamp(processed_frame, output_dir, "auto_frame")
        
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            if show_display:
                cv2.destroyAllWindows()
            
            print("Demo completed")
    
    def process_video_file(self, input_path: str, output_path: str):
        """
        Process a video file and save the result.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {input_path}")
        print(f"Total frames: {total_frames}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, gesture, confidence, current_fps = self.process_frame(frame)
                
                # Write frame
                out.write(processed_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Progress update every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        except Exception as e:
            print(f"Error processing video: {e}")
        
        finally:
            cap.release()
            out.release()
            print(f"Video processing completed: {output_path}")


def main():
    """Main inference function."""
    print("Starting Hand Tracking Inference...")
    
    parser = argparse.ArgumentParser(description='Hand tracking and gesture recognition demo')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--model-type', type=str, choices=['cnn', 'resnet', 'lightweight'],
                       default='lightweight', help='Model architecture type')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, default='assets', help='Output directory')
    parser.add_argument('--no-display', action='store_true', help='Disable live display')
    parser.add_argument('--save-frames', action='store_true', help='Save sample frames')
    parser.add_argument('--enable-mediapipe', action='store_true', help='Enable MediaPipe for hand detection')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed info')
    parser.add_argument('--trajectory', action='store_true', help='Show swipe trajectory visualization')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Camera ID: {args.camera}")
    print(f"  MediaPipe: {'Enabled' if args.enable_mediapipe else 'Disabled'}")
    print(f"  Debug mode: {args.debug}")
    print(f"  Trajectory: {args.trajectory}")
    
    # Create output directory
    try:
        os.makedirs(args.output, exist_ok=True)
        print(f"Output directory: {args.output}")
    except Exception as e:
        print(f"Warning: Could not create output directory: {e}")
    
    # Initialize inference
    try:
        print("Initializing inference system...")
        inference = HandTrackingInference(
            model_path=args.model,
            model_type=args.model_type,
            use_mediapipe=args.enable_mediapipe,
            debug_mode=args.debug,
            show_trajectory=args.trajectory
        )
        print("Inference system initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing inference system: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        if args.video:
            # Process video file
            output_video = os.path.join(args.output, f"processed_{os.path.basename(args.video)}")
            inference.process_video_file(args.video, output_video)
        else:
            # Run webcam demo
            inference.run_webcam_demo(
                camera_id=args.camera,
                output_dir=args.output,
                save_frames=args.save_frames,
                show_display=not args.no_display
            )
            
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
