"""Quick grip detection test - just hand detection, no pose"""
import sys
sys.path.insert(0, '.')

import cv2
import numpy as np

# Import MediaPipe
try:
    import importlib
    mp = importlib.import_module("mediapipe")
    _HAS_MEDIAPIPE = True
except Exception:
    mp = None
    _HAS_MEDIAPIPE = False

if not _HAS_MEDIAPIPE:
    print("ERROR: MediaPipe not installed")
    exit(1)

# Initialize hands detector
solutions = getattr(mp, "solutions", None)
hands_module = getattr(solutions, "hands", None) if solutions is not None else None
if hands_module is None:
    print("ERROR: MediaPipe hands not available")
    exit(1)

hands_detector = hands_module.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1,
)

print("Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

print("✓ Camera ready\n")
print("Show your hand to the camera. Alternate between OPEN and CLOSED fist.")
print("Watch the GripRaw values:\n")
print("  Open hand:  GripRaw should be ~0.35-0.45  → GripAngle 180-100°")
print("  Closed fist: GripRaw should be ~0.85-1.0 → GripAngle 0-30°\n")
print("=" * 60)

frame_count = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert and process
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_detector.process(rgb)
        
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            tip_idxs = [4, 8, 12, 16, 20]  # 5 fingertips
            wrist_lm = lm[0]  # wrist
            
            dists = []
            for idx in tip_idxs:
                dx = lm[idx].x - wrist_lm.x
                dy = lm[idx].y - wrist_lm.y
                dists.append((dx * dx + dy * dy) ** 0.5)
            
            avg = float(sum(dists) / len(dists))
            grip_angle = int(np.clip(np.interp(avg, [0.30, 1.00], [180, 0]), 0, 180))
            
            print(f"[Frame {frame_count}] GripRaw: {avg:.3f} → GripAngle: {grip_angle:3d}°")
        
        cv2.imshow("Grip Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Test ended")
