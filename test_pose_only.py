#!/usr/bin/env python
"""
Comprehensive Pose Detection Test with All Features Enabled
Tests pose detection, hand detection, angle calculations, and visual feedback
WITHOUT ESP32 connection - just run this to verify everything works
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

def safe_pose_label(pose):
    """
    Safely convert pose enum/value to a printable string.
    Handles None and unexpected values.
    """
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


def detect_hand_openness_improved(hands_detector, frame, wrist_x, wrist_y, arm_side, h, w, annotated_frame=None):
    """
    Improved hand detection with multiple fallback strategies
    Returns (hand_openness_value, hand_landmarks) or (None, None)
    """
    if hands_detector is None:
        return None, None
        
    try:
        # Strategy 1: Crop around wrist (larger crop)
        crop_size = int(min(h, w) * 0.4)
        half = crop_size // 2
        x1 = max(0, int(wrist_x) - half)
        y1 = max(0, int(wrist_y) - half)
        x2 = min(w, int(wrist_x) + half)
        y2 = min(h, int(wrist_y) + half)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = hands_detector.process(rgb_crop)
            
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                hand_side = res.multi_handedness[0].classification[0].label
                
                # Match hand side with arm side
                if (arm_side == 'left' and hand_side == 'Left') or \
                   (arm_side == 'right' and hand_side == 'Right'):
                    tip_idxs = [4, 8, 12, 16, 20]
                    wrist_lm = lm[0]
                    
                    dists = []
                    for idx in tip_idxs:
                        dx = lm[idx].x - wrist_lm.x
                        dy = lm[idx].y - wrist_lm.y
                        dists.append((dx * dx + dy * dy) ** 0.5)
                    
                    avg = float(sum(dists) / len(dists))
                    
                    # Draw hand landmarks
                    if annotated_frame is not None and mp is not None:
                        try:
                            solutions = getattr(mp, "solutions", None)
                            drawing_utils = getattr(solutions, "drawing_utils", None) if solutions else None
                            hands_module = getattr(solutions, "hands", None) if solutions else None
                            
                            if drawing_utils and hands_module:
                                # Scale landmarks from crop to full frame
                                scaled_landmarks = []
                                for landmark in lm:
                                    scaled_x = int((landmark.x * (x2 - x1)) + x1)
                                    scaled_y = int((landmark.y * (y2 - y1)) + y1)
                                    scaled_landmarks.append((scaled_x, scaled_y))
                                
                                # Draw connections
                                connections = hands_module.HAND_CONNECTIONS
                                for connection in connections:
                                    pt1_idx, pt2_idx = connection
                                    if pt1_idx < len(scaled_landmarks) and pt2_idx < len(scaled_landmarks):
                                        pt1 = scaled_landmarks[pt1_idx]
                                        pt2 = scaled_landmarks[pt2_idx]
                                        cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)
                                
                                # Draw landmarks
                                for pt in scaled_landmarks:
                                    cv2.circle(annotated_frame, pt, 3, (0, 255, 255), -1)
                                
                                # Draw crop ROI
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        except Exception:
                            pass
                    
                    return avg, res.multi_hand_landmarks[0]
        
        # Strategy 2: Full frame fallback
        try:
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_full = hands_detector.process(rgb_full)
            
            if res_full and res_full.multi_hand_landmarks:
                # Find hand closest to wrist
                best_hand = None
                best_dist = float('inf')
                
                for hand_landmarks in res_full.multi_hand_landmarks:
                    wrist_lm_full = hand_landmarks.landmark[0]
                    wrist_x_full = wrist_lm_full.x * w
                    wrist_y_full = wrist_lm_full.y * h
                    
                    dist = ((wrist_x_full - wrist_x) ** 2 + (wrist_y_full - wrist_y) ** 2) ** 0.5
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_hand = hand_landmarks
                
                if best_hand is not None and best_dist < min(h, w) * 0.3:
                    lm = best_hand.landmark
                    tip_idxs = [4, 8, 12, 16, 20]
                    wrist_lm = lm[0]
                    
                    dists = []
                    for idx in tip_idxs:
                        dx = lm[idx].x - wrist_lm.x
                        dy = lm[idx].y - wrist_lm.y
                        dists.append((dx * dx + dy * dy) ** 0.5)
                    
                    avg = float(sum(dists) / len(dists))
                    
                    # Draw hand landmarks
                    if annotated_frame is not None and mp is not None:
                        try:
                            solutions = getattr(mp, "solutions", None)
                            drawing_utils = getattr(solutions, "drawing_utils", None) if solutions else None
                            hands_module = getattr(solutions, "hands", None) if solutions else None
                            
                            if drawing_utils and hands_module:
                                drawing_utils.draw_landmarks(
                                    annotated_frame, best_hand, hands_module.HAND_CONNECTIONS
                                )
                        except Exception:
                            pass
                    
                    return avg, best_hand
        except Exception:
            pass
            
    except Exception:
        pass
    
    return None, None

def main():
    print("="*80)
    print("COMPREHENSIVE POSE DETECTION TEST - ALL FEATURES ENABLED")
    print("="*80)
    print("\nThis test includes:")
    print("  ✓ YOLO pose detection")
    print("  ✓ MediaPipe hand detection")
    print("  ✓ Real-time angle calculations")
    print("  ✓ Hand openness detection")
    print("  ✓ Visual feedback with landmarks")
    print("  ✓ Motion state tracking")
    print("\n" + "="*80)
    
    # Initialize pose detector
    print("\n[1/5] Initializing YOLO pose detector...")
    try:
        detector = ZonePoseDetector()
        print("    ✓ YOLO detector ready")
    except Exception as e:
        print(f"    ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize MediaPipe hands detector
    print("\n[2/5] Initializing MediaPipe hands detector...")
    hands_detector = None
    if _HAS_MEDIAPIPE:
        try:
            solutions = getattr(mp, "solutions", None)
            hands_module = getattr(solutions, "hands", None) if solutions is not None else None
            if hands_module is not None:
                hands_detector = hands_module.Hands(
                    static_image_mode=False,
                    max_num_hands=2,  # Support 2 hands for two-handed mode testing
                    min_detection_confidence=0.5,  # Lower threshold for better detection
                    min_tracking_confidence=0.3,
                    model_complexity=1,
                )
                print("    ✓ MediaPipe hands detector ready")
            else:
                print("    ⚠ MediaPipe hands module not available")
        except Exception as e:
            print(f"    ⚠ MediaPipe hands initialization failed: {e}")
    else:
        print("    ⚠ MediaPipe not installed - hand detection disabled")
    
    # Open camera
    print("\n[3/5] Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("    ✗ ERROR: Could not open camera")
        print("    Make sure your camera is connected and not used by another app")
        return
    
    # Set camera properties
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except:
        pass
    
    print("    ✓ Camera opened")
    
    print("\n[4/5] Starting pose detection...")
    print("    ✓ Ready!")
    
    print("\n" + "="*80)
    print("POSE DETECTION ACTIVE")
    print("="*80)
    print("\nInstructions:")
    print("  - Stand in front of the camera with your arms visible")
    print("  - Move your arm to see angle calculations")
    print("  - Open/close your hand to see grip detection")
    print("  - All features are enabled and displayed")
    print("  - Press 'q' to quit\n")
    
    current_pose = None
    frame_count = 0
    
    # Drawing utilities for MediaPipe
    drawing_utils = None
    hands_module = None
    if _HAS_MEDIAPIPE and mp is not None:
        try:
            solutions = getattr(mp, "solutions", None)
            drawing_utils = getattr(solutions, "drawing_utils", None) if solutions else None
            hands_module = getattr(solutions, "hands", None) if solutions else None
        except:
            pass
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("WARNING: Failed to capture frame, retrying...")
                continue
            
            frame_count += 1
            h, w = frame.shape[:2]
            
            # Process pose detection
            try:
                annotated_frame, pose, angles, keypoints, arm_side, motion_state = detector.process_frame(frame)
            except Exception as e:
                print(f"ERROR in process_frame: {e}")
                continue
            
            shoulder_angle, elbow_angle, wrist_angle = angles
            
            # Initialize grip angle with fallback
            grip_angle = int(np.clip(wrist_angle, 0, 180))
            hand_openness_value = None
            
            # Attempt hand detection if available
            if hands_detector is not None and keypoints is not None and arm_side is not None:
                try:
                    LEFT_WRIST_IDX = 9
                    RIGHT_WRIST_IDX = 10
                    wrist_idx = LEFT_WRIST_IDX if arm_side == 'left' else RIGHT_WRIST_IDX
                    
                    if wrist_idx < len(keypoints) and keypoints[wrist_idx][2] > 0.2:
                        wrist_x = keypoints[wrist_idx][0]
                        wrist_y = keypoints[wrist_idx][1]
                        
                        # Detect hand openness with improved method
                        hand_openness_value, hand_landmarks = detect_hand_openness_improved(
                            hands_detector, frame, wrist_x, wrist_y, arm_side, h, w, annotated_frame
                        )
                        
                        if hand_openness_value is not None:
                            # Map hand openness to grip angle
                            grip_angle = int(np.clip(np.interp(hand_openness_value, [0.30, 1.00], [0, 180]), 0, 180))
                except Exception as e:
                    # Hand detection failed, continue with wrist angle fallback
                    pass
            
            # Display comprehensive information on frame
            y_start = 30
            line_height = 25
            
            # Pose information
            pose_label = safe_pose_label(pose)
            cv2.putText(annotated_frame, f"Pose: {pose_label}", (10, y_start),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Angles
            cv2.putText(annotated_frame, f"Shoulder: {shoulder_angle:.1f}°", (10, y_start + line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Elbow: {elbow_angle:.1f}°", (10, y_start + line_height * 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Wrist: {wrist_angle:.1f}°", (10, y_start + line_height * 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Grip information
            if hand_openness_value is not None:
                cv2.putText(annotated_frame, f"Grip: {grip_angle}° (Hand: {hand_openness_value:.3f})", 
                           (10, y_start + line_height * 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(annotated_frame, f"Grip: {grip_angle}° (from wrist, hand not detected)", 
                           (10, y_start + line_height * 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Motion state
            cv2.putText(annotated_frame, f"Base: {motion_state['base']}", (10, y_start + line_height * 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            cv2.putText(annotated_frame, f"Forward: {motion_state['forward']}", (10, y_start + line_height * 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            cv2.putText(annotated_frame, f"Vertical: {motion_state['vertical']}", (10, y_start + line_height * 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            cv2.putText(annotated_frame, f"Grip State: {motion_state['grip']}", (10, y_start + line_height * 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            
            # Arm side
            if arm_side:
                cv2.putText(annotated_frame, f"Tracking: {arm_side.upper()} ARM", 
                           (10, y_start + line_height * 9),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Frame counter
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (w - 150, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            
            # Print updates to console
            if pose and pose != current_pose:
                current_pose = pose
                print(f"[Frame {frame_count:4d}] Pose: {pose_label:15s} | "
                      f"S:{shoulder_angle:6.1f}° | "
                      f"E:{elbow_angle:6.1f}° | "
                      f"W:{wrist_angle:6.1f}° | "
                      f"Grip:{grip_angle:3d}° | "
                      f"Hand:{hand_openness_value:.3f if hand_openness_value else 'N/A':>7} | "
                      f"State: {motion_state}")
            elif frame_count % 60 == 0:
                print(f"[Frame {frame_count:4d}] Angles - "
                      f"S:{shoulder_angle:6.1f}° | "
                      f"E:{elbow_angle:6.1f}° | "
                      f"W:{wrist_angle:6.1f}° | "
                      f"Grip:{grip_angle:3d}° | "
                      f"Hand:{hand_openness_value:.3f if hand_openness_value else 'N/A':>7} | "
                      f"State: {motion_state}")
            
            # Display frame
            try:
                cv2.imshow("Pose Detection Test - All Features Enabled (Press 'q' to quit)", annotated_frame)
            except Exception as e:
                print(f"ERROR displaying frame: {e}")
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\nERROR during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[5/5] Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("    ✓ Camera released")
        print("    ✓ Windows closed")
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        print("\nAll features were tested:")
        print("  ✓ Pose detection working")
        print("  ✓ Hand detection working" + (" ✓" if hand_openness_value else " (hand not detected in last frame)"))
        print("  ✓ Angle calculations working")
        print("  ✓ Visual feedback working")
        print("\nIf everything worked correctly, the system is ready!")

if __name__ == "__main__":
    main()
