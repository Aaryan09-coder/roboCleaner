#!/usr/bin/env python
"""
Minimal OpenCV test to diagnose camera issues
"""
import cv2
import sys

print("Testing OpenCV version:", cv2.__version__)
print("\nAttempting to open camera...")

# Try to open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not opened")
    print("Trying camera index 1...")
    cap = cv2.VideoCapture(1)
    
if not cap.isOpened():
    print("ERROR: Could not open any camera")
    print("\nPossible issues:")
    print("  1. Camera not connected")
    print("  2. Camera being used by another app")
    print("  3. Camera permissions not granted")
    sys.exit(1)

print("✓ Camera opened!")

# Try to read a frame
print("\nAttempting to read frame...")
ret, frame = cap.read()

if not ret:
    print("ERROR: Could not read frame")
    cap.release()
    sys.exit(1)

if frame is None:
    print("ERROR: Frame is None")
    cap.release()
    sys.exit(1)

print(f"✓ Frame captured! Shape: {frame.shape}")

# Try to display
print("\nAttempting to display frame...")
try:
    cv2.imshow("Test Window", frame)
    print("✓ Window created")
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✓ Window closed")
except Exception as e:
    print(f"ERROR displaying: {e}")
    import traceback
    traceback.print_exc()

cap.release()
print("\n✓ Test complete!")

