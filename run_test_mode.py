#!/usr/bin/env python
"""Simple test mode - just run pose detection without ESP32"""
import sys
import os
sys.path.insert(0, '.')

from ml_model.yolo_fightingpose_detection import ZonePoseDetector
import cv2

print("="*60)
print("Nuclear Waste Cleaning Arm - Test Mode")
print("="*60)
print("\nInitializing pose detector...")
detector = ZonePoseDetector()
print("✓ Detector ready")

print("\nOpening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

print("✓ Camera opened")
print("\n" + "="*60)
print("POSE DETECTION ACTIVE")
print("="*60)
print("Move your arm to see real-time angle calculations")
print("Press 'q' in the window to quit\n")

current_pose = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to capture frame")
            break
        
        annotated_frame, pose, angles = detector.process_frame(frame)
        
        if pose and pose != current_pose:
            current_pose = pose
            print(f"Pose: {pose.value} | Shoulder: {angles[0]:.1f}° | Elbow: {angles[1]:.1f}° | Wrist: {angles[2]:.1f}°")
        
        cv2.imshow("Nuclear Waste Cleaning Arm Control", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nQuitting...")
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Done")

