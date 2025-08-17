#!/bin/bash
# Hand Tracking Demo Runner
# This script automatically uses the correct Python environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python"

echo "Hand Tracking Gesture Recognition"
echo "================================="

# Check if virtual environment exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Virtual environment not found at $PYTHON_PATH"
    echo "Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo "Using Python: $PYTHON_PATH"
echo ""

# Parse command line arguments
case "$1" in
    "demo"|"")
        echo "Running basic demo..."
        "$PYTHON_PATH" src/inference.py
        ;;
    "debug")
        echo "Running debug mode..."
        "$PYTHON_PATH" src/inference.py --debug
        ;;
    "trajectory")
        echo "Running with trajectory visualization..."
        "$PYTHON_PATH" src/inference.py --trajectory
        ;;
    "debug-trajectory")
        echo "Running debug mode with trajectory..."
        "$PYTHON_PATH" src/inference.py --debug --trajectory
        ;;
    "no-mediapipe")
        echo "Running without MediaPipe..."
        "$PYTHON_PATH" src/inference.py --debug
        ;;
    "mediapipe")
        echo "Running with MediaPipe enabled..."
        "$PYTHON_PATH" src/inference.py --enable-mediapipe --debug
        ;;
    "train")
        echo "Training model..."
        "$PYTHON_PATH" src/train.py
        ;;
    "test")
        echo "Testing camera and system..."
        "$PYTHON_PATH" test_camera.py
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  demo              - Run basic webcam demo (default)"
        echo "  debug             - Run with debug information"
        echo "  trajectory        - Run with trajectory visualization"
        echo "  debug-trajectory  - Run debug mode with trajectory"
        echo "  no-mediapipe      - Run without MediaPipe (faster startup)"
        echo "  train             - Train a new model"
        echo "  test              - Test camera and system components"
        echo "  help              - Show this help message"
        echo ""
        echo "Example:"
        echo "  $0 debug"
        echo "  $0 trajectory"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for available commands"
        exit 1
        ;;
esac
