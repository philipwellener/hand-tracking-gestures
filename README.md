# Hand Tracking Gestures

A Python project I built to explore real-time hand tracking and gesture recognition using MediaPipe and PyTorch, combining computer vision with neural networks for intuitive human-computer interaction.

![Demo](assets/demo.gif)

*Note: The GIF compression makes the visualization appear choppy - the actual real-time performance is much smoother*

## What it does

- Detects and tracks hand landmarks in real-time using MediaPipe
- Recognizes static gestures like thumbs up, peace sign, fist, open palm, pointing, pinch, and OK sign
- Detects dynamic swipe gestures in four directions with trajectory visualization
- Uses a lightweight PyTorch neural network for fast gesture classification
- Provides interactive webcam demo with visual feedback and debug information

## Project structure

```
hand-tracking-gestures/
├── src/
│   ├── dataset.py         # dataset loading and preprocessing utilities
│   ├── gestures.py        # gesture detection algorithms and swipe logic
│   ├── inference.py       # real-time inference engine and webcam demo
│   ├── model.py          # lightweight PyTorch neural network architecture
│   ├── train.py          # model training pipeline with validation
│   ├── utils.py          # utility functions and data processing
│   └── visualize.py      # visualization tools and trajectory drawing
├── checkpoints/          # trained model weights and configurations
├── runs/                 # training logs and tensorboard metrics
├── assets/              # demo gifs and documentation media
├── run.sh              # convenience script for running demos
├── requirements.txt     # python dependencies
├── .gitignore          # git ignore patterns
└── README.md           # you are here
```

## Building and Running

```bash
# install dependencies (if not already done)
pip install -r requirements.txt

# ALWAYS activate virtual environment first
source .venv/bin/activate

# run the main webcam demo
python src/inference.py

# run with debug visualization
python src/inference.py --debug

# train a new model
python src/train.py
```

**Alternative: Use full Python path (if activation doesn't work):**
```bash
# run the main webcam demo
./.venv/bin/python src/inference.py

# run with debug visualization
./.venv/bin/python src/inference.py --debug

# run with trajectory visualization
./.venv/bin/python src/inference.py --trajectory
```

For development and testing:
```bash
# run inference with trajectory visualization
python src/inference.py --trajectory

# train with custom parameters
python src/train.py --epochs 100 --batch-size 64

# evaluate model performance
python src/train.py --evaluate-only
```

**CRITICAL:** Always run `source .venv/bin/activate` first, or use the full Python path!

## How to use it

Point your webcam at your hand and try different gestures. The system recognizes:

**Static Gestures:**
- Thumbs up (thumb extended upward)
- Peace sign (index and middle fingers in V shape)
- Fist (closed hand)
- Open palm (all fingers extended)
- Pointing (index finger extended)
- Pinch (thumb and index finger close together)
- OK sign (thumb and index forming circle)

**Dynamic Gestures:**
- Swipe left/right (horizontal hand movement)
- Swipe up/down (vertical hand movement)

The trajectory visualization shows your hand movement path with green start points, red end points, and a fading trail.

## The technical bits

### Hand Tracking
MediaPipe provides robust hand landmark detection:
- 21 3D hand landmarks per frame
- Real-time processing at 30+ FPS
- Normalized coordinates for scale invariance
- Reliable tracking across lighting conditions

### Gesture Classification
Lightweight PyTorch neural network for static gesture recognition:
- Input: 42 features (21 landmarks x 2D coordinates)
- Hidden layers with batch normalization and dropout
- Softmax output for gesture probability distribution
- Fast inference suitable for real-time applications

### Swipe Detection
Dynamic gesture recognition using trajectory analysis:
- Tracks index fingertip movement over time
- Calculates movement distance and velocity
- Direction classification based on dominant movement axis
- Cooldown mechanism to prevent gesture spam

### Visualization System
Comprehensive visual feedback:
- Hand skeleton drawing with color-coded fingers
- Real-time trajectory paths with fading effects
- Gesture confidence scores and FPS counters
- Debug information for movement statistics

## Recent Updates
Been refining the gesture detection based on testing:
- Improved swipe detection with better trajectory filtering
- Enhanced visualization with movement statistics and debug info
- More robust gesture classification with confidence thresholds
- Added trajectory cooldown to reduce false positive swipes
- Better hand landmark drawing with finger color coding
- Integrated debug functionality into main inference pipeline

The system is modular - you can easily add new gestures by extending the classification model or detection algorithms.

## Lessons learned

### What went well:
- MediaPipe hand tracking worked incredibly well out of the box
- Lightweight PyTorch neural network achieved high accuracy with minimal training
- Real-time performance was better than expected
- Modular design made adding new gestures straightforward

### What was trickier than expected:
- Swipe gesture detection needed careful trajectory filtering
- Balancing sensitivity vs noise in dynamic gesture recognition
- Coordinate normalization across different camera resolutions
- Managing gesture detection cooldowns to prevent spam

### If I did this again:
- Would implement gesture data collection tools first
- Add more sophisticated temporal modeling for dynamic gestures
- Better handling of partial hand occlusion scenarios
- Should have added gesture customization from the start

### Weird issues I ran into:
- Swipe detection would sometimes trigger on small hand tremors
- Lighting changes occasionally affected landmark stability
- Fast movements could cause landmark detection to lag
- Coordinate normalization behaved differently across devices

The biggest insight was that reliable gesture recognition requires more than just accurate hand tracking - temporal consistency, noise filtering, and robust trajectory analysis are equally important for a smooth user experience.

## TODO

- Add rotation and wave gesture detection (building on existing gesture set)
- Implement 3D gesture recognition using MediaPipe's Z-coordinate data
- Add gesture sequence recognition for complex commands
- Add gesture customization and user training modes
- Multi-hand tracking and two-handed gesture recognition
- Integration with applications for practical gesture control
