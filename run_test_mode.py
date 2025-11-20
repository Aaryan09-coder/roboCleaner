#!/usr/bin/env python
"""Simple test mode - just run pose detection without ESP32"""
import sys
import os
sys.path.insert(0, '.')

from ml_model.yolo_fightingpose_detection import ZonePoseDetector
import cv2
import numpy as np

# Import MediaPipe for hand detection
try:
    import importlib
    mp = importlib.import_module("mediapipe")
    _HAS_MEDIAPIPE = True
except Exception:
    mp = None
    _HAS_MEDIAPIPE = False

# Helper to safely display pose values
def safe_pose_label(pose):
    try:
        if pose is None:
            return "UNKNOWN"
        if hasattr(pose, "value"):
            label = pose.value
        else:
            label = pose
        if label is None:
            return "UNKNOWN"
        return str(label)
    except Exception:
        return "UNKNOWN"

print("="*60)
print("Nuclear Waste Cleaning Arm - Test Mode")
print("="*60)
print("\nInitializing pose detector...")
detector = ZonePoseDetector()
print("✓ Detector ready")

# Initialize MediaPipe hands if available
hands_detector = None
if _HAS_MEDIAPIPE:
    try:
        solutions = getattr(mp, "solutions", None)
        hands_module = getattr(solutions, "hands", None) if solutions is not None else None
        if hands_module is not None:
            hands_detector = hands_module.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                model_complexity=1,
            )
            print("✓ MediaPipe hands initialized")
    except Exception as e:
        print(f"⚠ MediaPipe hands unavailable: {e}")

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
        
        annotated_frame, pose, angles, keypoints, arm_side, motion_state = detector.process_frame(frame)
        
        # Extract angles
        shoulder_angle, elbow_angle, wrist_angle = angles
        grip_angle = int(np.clip(wrist_angle, 0, 180))  # default fallback
        
        # Try to detect hand for grip
        if hands_detector is not None and keypoints is not None and arm_side is not None:
            try:
                LEFT_WRIST_IDX = 9
                RIGHT_WRIST_IDX = 10
                wrist_idx = LEFT_WRIST_IDX if arm_side == 'left' else RIGHT_WRIST_IDX
                
                if keypoints is not None and wrist_idx < len(keypoints) and keypoints[wrist_idx][2] > 0.2:
                    print(f"[DEBUG] Wrist detected: conf={keypoints[wrist_idx][2]:.2f}")
                    wrist_xy = keypoints[wrist_idx][:2].astype(int)
                    h, w, _ = frame.shape
                    cx, cy = int(wrist_xy[0]), int(wrist_xy[1])
                    crop_size = int(min(h, w) * 0.25)
                    half = crop_size // 2
                    x1 = max(0, cx - half)
                    y1 = max(0, cy - half)
                    x2 = min(w, cx + half)
                    y2 = min(h, cy + half)
                    print(f"[DEBUG] Crop ROI: ({x1},{y1}) to ({x2},{y2}), size={crop_size}x{crop_size}")
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (200, 200, 0), 2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        res = hands_detector.process(rgb_crop)
                        if res.multi_hand_landmarks:
                            print(f"[DEBUG] Hand detected! Landmarks: {len(res.multi_hand_landmarks[0].landmark)}")
                            lm = res.multi_hand_landmarks[0].landmark
                            tip_idxs = [4, 8, 12, 16, 20]
                            wrist_lm = lm[0]
                            dists = []
                            for idx in tip_idxs:
                                dx = lm[idx].x - wrist_lm.x
                                dy = lm[idx].y - wrist_lm.y
                                dists.append((dx * dx + dy * dy) ** 0.5)
                            avg = float(sum(dists) / len(dists))
                            # Map: closed (large distances) -> high angle (180), open (small distances) -> low angle (0)
                            # Calibrated for your camera: closed ~0.85-1.0, open ~0.35-0.45
                            grip_angle = int(np.clip(np.interp(avg, [0.30, 1.00], [180, 0]), 0, 180))
                            print(f"[DEBUG] GripRaw: {avg:.3f} -> GripAngle: {grip_angle}°")
                            cv2.putText(annotated_frame, f"GripRaw: {avg:.3f}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                        else:
                            print(f"[DEBUG] No hand landmarks detected in crop")
                    else:
                        print(f"[DEBUG] Empty crop")
                else:
                    print(f"[DEBUG] Wrist keypoint not available or low confidence")
            except Exception as e:
                print(f"[DEBUG] Hand detection exception: {e}")
        
        if pose and pose != current_pose:
            current_pose = pose
            pose_label = safe_pose_label(pose)
            print(f"Pose: {pose_label} | Shoulder: {shoulder_angle:.1f}° | Elbow: {elbow_angle:.1f}° | Wrist: {wrist_angle:.1f}° | Grip: {grip_angle}° | Motion: {motion_state}")
        
        cv2.imshow("Nuclear Waste Cleaning Arm Control", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nQuitting...")
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Done")


