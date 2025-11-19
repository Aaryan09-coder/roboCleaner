#!/usr/bin/env python
"""
Full end-to-end execution of the Nuclear Waste Cleaning Arm system
This script runs the complete system with pose detection and arm control
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("="*60)
    print("NUCLEAR WASTE CLEANING ARM - FULL SYSTEM")
    print("="*60)
    print("\nChoose execution mode:")
    print("1. Pose Detection Only (YOLO mode - camera control)")
    print("2. Keyboard Control Mode")
    print("3. Test Pose Detection (no ESP32 connection)")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n" + "="*60)
        print("Starting Pose Detection Mode")
        print("="*60)
        print("This mode requires ESP32 connection.")
        print("Make sure ESP32 is powered and connected to WiFi.")
        print("Press 'q' in camera window to quit.\n")
        
        try:
            from client.PoseController import PoseController
            controller = PoseController()
            if controller.connect():
                print("✓ Connected to ESP32")
                controller.run_yolo_mode()
            else:
                print("⚠ Failed to connect to ESP32")
                print("Running in test mode (pose detection only)...")
                from ml_model.yolo_fightingpose_detection import ZonePoseDetector
                import cv2
                
                detector = ZonePoseDetector()
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    print("ERROR: Could not open camera")
                    return
                
                print("Camera opened. Starting pose detection...")
                current_pose = None
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        annotated_frame, pose, angles, *_ = detector.process_frame(frame)
                        
                        if pose and pose != current_pose:
                            current_pose = pose
                            print(f"Pose: {pose.value} | Angles - Shoulder: {angles[0]:.1f}°, Elbow: {angles[1]:.1f}°, Wrist: {angles[2]:.1f}°")
                        
                        cv2.imshow("Nuclear Waste Cleaning Arm Control", annotated_frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
                    detector.reset_arm() if hasattr(detector, 'reset_arm') else None
                    
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "2":
        print("\n" + "="*60)
        print("Starting Keyboard Control Mode")
        print("="*60)
        print("This mode requires ESP32 connection.")
        print("Controls:")
        print("  A/D - Base left/right")
        print("  W/S - Forward/backward")
        print("  Q/E - Up/down")
        print("  Z/X - Grip open/close")
        print("  R   - Reset arm")
        print("  ESC - Exit\n")
        
        try:
            from client.KeyboardController import KeyboardController
            controller = KeyboardController()
            if controller.connect():
                print("✓ Connected to ESP32")
                controller.run_keyboard_mode()
            else:
                print("ERROR: Failed to connect to ESP32")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "3":
        print("\n" + "="*60)
        print("Test Mode - Pose Detection Only")
        print("="*60)
        print("No ESP32 connection required.")
        print("This will test the pose detection system.")
        print("Press 'q' in camera window to quit.\n")
        
        try:
            from ml_model.yolo_fightingpose_detection import ZonePoseDetector
            import cv2
            import importlib
            
            # Try to initialize MediaPipe hands for grip detection (optional)
            try:
                mp = importlib.import_module("mediapipe")
                _HAS_MEDIAPIPE = True
            except Exception:
                mp = None
                _HAS_MEDIAPIPE = False

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
                except Exception:
                    hands_detector = None

            print("Initializing detector...")
            detector = ZonePoseDetector()
            print("✓ Detector initialized")

            print("Opening camera...")
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("ERROR: Could not open camera")
                return

            print("✓ Camera opened")
            print("\nStarting pose detection...")
            print("Move your arm to see angle calculations")
            print("Press 'q' to quit\n")

            current_pose = None
            frame_count = 0

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("ERROR: Failed to capture frame")
                        break

                    frame_count += 1
                    # Unpack keypoints and arm_side so we can run hand detection for grip
                    annotated_frame, pose, angles, keypoints, arm_side, motion_state = detector.process_frame(frame)

                    # Default grip fallback uses wrist angle
                    import numpy as _np
                    shoulder_angle, elbow_angle, wrist_angle = angles
                    grip_angle = int(_np.clip(wrist_angle, 0, 180))

                    # Attempt MediaPipe hand detection if available
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
                                    used_landmarks = None
                                    if res.multi_hand_landmarks:
                                        print(f"[DEBUG] Hand detected in crop! Landmarks: {len(res.multi_hand_landmarks[0].landmark)}")
                                        used_landmarks = res.multi_hand_landmarks[0].landmark
                                    else:
                                        # Fallback: try running detector on full frame if crop fails
                                        print(f"[DEBUG] No hand landmarks detected in crop — trying full frame fallback")
                                        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        res_full = hands_detector.process(rgb_full)
                                        if res_full and res_full.multi_hand_landmarks:
                                            print(f"[DEBUG] Hand detected on full frame! Landmarks: {len(res_full.multi_hand_landmarks[0].landmark)}")
                                            used_landmarks = res_full.multi_hand_landmarks[0].landmark

                                    if used_landmarks is not None:
                                        lm = used_landmarks
                                        tip_idxs = [4, 8, 12, 16, 20]
                                        wrist_lm = lm[0]
                                        dists = []
                                        for idx in tip_idxs:
                                            dx = lm[idx].x - wrist_lm.x
                                            dy = lm[idx].y - wrist_lm.y
                                            dists.append((dx * dx + dy * dy) ** 0.5)
                                        avg = float(sum(dists) / len(dists))
                                        # Calibrated mapping for this camera: closed ~0.85-1.0, open ~0.35-0.45
                                        grip_angle = int(_np.clip(_np.interp(avg, [0.30, 1.00], [180, 0]), 0, 180))
                                        print(f"[DEBUG] GripRaw: {avg:.3f} -> GripAngle: {grip_angle}°")
                                        cv2.putText(annotated_frame, f"GripRaw: {avg:.3f}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                                    else:
                                        print(f"[DEBUG] No hand landmarks detected in crop or full frame")
                                else:
                                    print(f"[DEBUG] Empty crop")
                            else:
                                print(f"[DEBUG] Wrist keypoint not available or low confidence")
                        except Exception as e:
                            print(f"[DEBUG] Hand detection exception: {e}")
                            pass

                    # Print updates every 30 frames or when pose changes
                    if pose and pose != current_pose:
                        current_pose = pose
                        print(f"[Frame {frame_count}] Pose: {pose.value} | Motion: {motion_state}")
                        print(f"  Shoulder: {angles[0]:.1f}° | Elbow: {angles[1]:.1f}° | Wrist: {angles[2]:.1f}° | Grip: {grip_angle}")
                    elif frame_count % 30 == 0:
                        print(f"[Frame {frame_count}] Angles - Shoulder: {angles[0]:.1f}° | Elbow: {angles[1]:.1f}° | Wrist: {angles[2]:.1f}° | Grip: {grip_angle} | Motion: {motion_state}")

                    cv2.imshow("Nuclear Waste Cleaning Arm Control - Test Mode", annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nQuitting...")
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()
                print("✓ Test completed")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "4":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Please run again and select 1-4.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()

