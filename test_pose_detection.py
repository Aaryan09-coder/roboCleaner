#!/usr/bin/env python
"""Test script to diagnose pose detection issues"""
import sys
import traceback

print("=" * 50)
print("Testing Pose Detection System")
print("=" * 50)

# Test 1: Check Python version
print(f"\n1. Python version: {sys.version}")

# Test 2: Check NumPy
try:
    import numpy
    print(f"2. NumPy version: {numpy.__version__}")
    print(f"   NumPy location: {numpy.__file__}")
except Exception as e:
    print(f"2. NumPy import failed: {e}")
    traceback.print_exc()

# Test 3: Check torch
try:
    import torch
    print(f"3. PyTorch version: {torch.__version__}")
except Exception as e:
    print(f"3. PyTorch import failed: {e}")
    traceback.print_exc()

# Test 4: Check ultralytics
try:
    from ultralytics import YOLO
    print("4. Ultralytics import: SUCCESS")
except Exception as e:
    print(f"4. Ultralytics import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check OpenCV
try:
    import cv2
    print(f"5. OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"5. OpenCV import failed: {e}")
    traceback.print_exc()

# Test 6: Import detector
try:
    sys.path.insert(0, '.')
    from ml_model.yolo_fightingpose_detection import ZonePoseDetector
    print("6. ZonePoseDetector import: SUCCESS")
except Exception as e:
    print(f"6. ZonePoseDetector import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: Initialize detector
try:
    print("\n7. Initializing detector...")
    detector = ZonePoseDetector()
    print("   Detector initialized: SUCCESS")
except Exception as e:
    print(f"7. Detector initialization failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 8: Check camera
try:
    print("\n8. Testing camera...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("   Camera opened: SUCCESS")
        ret, frame = cap.read()
        if ret:
            print(f"   Frame captured: SUCCESS (shape: {frame.shape})")
        else:
            print("   Frame capture: FAILED")
        cap.release()
    else:
        print("   Camera opened: FAILED")
except Exception as e:
    print(f"8. Camera test failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 50)
print("All tests completed!")
print("=" * 50)

