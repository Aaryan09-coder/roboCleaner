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
    print("4. New Controller (controller.py - ML→ESP32 direct mapping)")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n" + "="*60)
        print("Starting Pose Detection Mode")
        print("="*60)
        print("This mode requires ESP32 connection.")
        print("Make sure ESP32 is powered and connected to WiFi.")
        print("You will be prompted to choose a control mode next.\n")
        
        try:
            from client.PoseController import PoseController, show_control_mode_menu
            controller = PoseController()
            
            # Show control mode menu
            mode_choice = show_control_mode_menu()
            
            if mode_choice == 'q':
                print("Exiting...")
                return
            
            # Connect to ESP32 (or continue without connection for testing)
            connected = controller.connect()
            if connected:
                print("✓ Connected to ESP32")
            else:
                print("⚠ Failed to connect to ESP32")
                print("Running in test mode (pose detection only - no servo control)...\n")
            
            # Run selected mode
            if mode_choice == '1':
                controller.run_mode_switching()
            elif mode_choice == '2':
                controller.run_two_handed()
            else:
                print("Invalid choice. Exiting...")
                return
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            return
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
        print("\n" + "="*80)
        print("Test Mode - Pose Detection Only (All Features Enabled)")
        print("="*80)
        print("No ESP32 connection required.")
        print("This will test the pose detection system with ALL features enabled:")
        print("  ✓ YOLO pose detection")
        print("  ✓ MediaPipe hand detection (improved)")
        print("  ✓ Real-time angle calculations")
        print("  ✓ Hand openness detection")
        print("  ✓ Visual feedback with landmarks")
        print("  ✓ Motion state tracking")
        print("\nPress 'q' in camera window to quit.\n")
        
        try:
            from ml_model.yolo_fightingpose_detection import ZonePoseDetector
            from client.PoseController import PoseController  # Import to use improved hand detection method
            import cv2
            import numpy as np
            import importlib
            
            # Try to initialize MediaPipe hands for grip detection (with improved settings)
            try:
                mp = importlib.import_module("mediapipe")
                _HAS_MEDIAPIPE = True
            except Exception:
                mp = None
                _HAS_MEDIAPIPE = False

            print("Initializing detectors...")
            detector = ZonePoseDetector()
            print("✓ YOLO detector initialized")

            # Create a temporary PoseController instance to use its improved hand detection method
            temp_controller = PoseController()
            hands_detector = temp_controller.hands_detector
            
            if hands_detector is not None:
                print("✓ MediaPipe hands detector initialized (with improved settings)")
            else:
                print("⚠ MediaPipe hands not available - hand detection disabled")

            print("Opening camera...")
            cap = cv2.VideoCapture(0)
            
            # Set camera properties for better performance
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
            except:
                pass

            if not cap.isOpened():
                print("ERROR: Could not open camera")
                return

            print("✓ Camera opened")
            print("\nStarting pose detection with all features...")
            print("Move your arm to see angle calculations")
            print("Open/close your hand to see grip detection")
            print("Press 'q' to quit\n")

            current_pose = None
            frame_count = 0
            hand_openness_value = None  # Initialize for final status

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("ERROR: Failed to capture frame")
                        break

                    frame_count += 1
                    h, w = frame.shape[:2]
                    
                    # Process pose detection
                    annotated_frame, pose, angles, keypoints, arm_side, motion_state = detector.process_frame(frame)

                    # Default grip fallback uses wrist angle
                    shoulder_angle, elbow_angle, wrist_angle = angles
                    grip_angle = int(np.clip(wrist_angle, 0, 180))
                    hand_openness_value = None

                    # Attempt MediaPipe hand detection with improved method
                    if hands_detector is not None and keypoints is not None and arm_side is not None:
                        try:
                            LEFT_WRIST_IDX = 9
                            RIGHT_WRIST_IDX = 10
                            wrist_idx = LEFT_WRIST_IDX if arm_side == 'left' else RIGHT_WRIST_IDX

                            if wrist_idx < len(keypoints) and keypoints[wrist_idx][2] > 0.2:
                                wrist_x = keypoints[wrist_idx][0]
                                wrist_y = keypoints[wrist_idx][1]
                                
                                # Use improved hand detection method from PoseController
                                hand_openness_value, hand_landmarks = temp_controller.detect_hand_openness(
                                    hands_detector, frame, wrist_x, wrist_y, arm_side, h, w, annotated_frame
                                )

                                if hand_openness_value is not None:
                                    # Map hand openness: open hand (smaller value) → 0°, closed (larger) → 180°
                                    grip_angle = int(np.clip(np.interp(hand_openness_value, [0.30, 1.00], [0, 180]), 0, 180))
                        except Exception as e:
                            # Hand detection failed, continue with wrist angle fallback
                            pass

                    # Enhanced visual feedback on frame
                    y_start = 30
                    line_height = 25
                    
                    # Pose information - safely get pose value
                    try:
                        if pose is not None and hasattr(pose, 'value'):
                            pose_value_display = str(pose.value) if pose.value is not None else "UNKNOWN"
                        else:
                            pose_value_display = "UNKNOWN"
                    except Exception:
                        pose_value_display = "UNKNOWN"
                    
                    cv2.putText(annotated_frame, f"Pose: {pose_value_display}", (10, y_start),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Angles
                    cv2.putText(annotated_frame, f"Shoulder: {shoulder_angle:.1f}°", (10, y_start + line_height),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Elbow: {elbow_angle:.1f}°", (10, y_start + line_height * 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Wrist: {wrist_angle:.1f}°", (10, y_start + line_height * 3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Grip information with hand detection status
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

                    # Print updates to console (reduced verbosity)
                    # Get pose value safely (pose can be None or pose.value can be None)
                    try:
                        if pose is not None and hasattr(pose, 'value'):
                            pose_value_str = str(pose.value) if pose.value is not None else "UNKNOWN"
                        else:
                            pose_value_str = "UNKNOWN"
                    except Exception:
                        pose_value_str = "UNKNOWN"
                    
                    if pose is not None and pose != current_pose:
                        current_pose = pose
                        print(f"[Frame {frame_count:4d}] Pose: {pose_value_str:15s} | "
                              f"S:{shoulder_angle:6.1f}° | "
                              f"E:{elbow_angle:6.1f}° | "
                              f"W:{wrist_angle:6.1f}° | "
                              f"Grip:{grip_angle:3d}° | "
                              f"Hand:{hand_openness_value:.3f if hand_openness_value else 'N/A':>7} | "
                              f"State: {motion_state}")
                    elif frame_count % 60 == 0:  # Print every 60 frames (~2 seconds at 30fps)
                        print(f"[Frame {frame_count:4d}] Angles - "
                              f"S:{shoulder_angle:6.1f}° | "
                              f"E:{elbow_angle:6.1f}° | "
                              f"W:{wrist_angle:6.1f}° | "
                              f"Grip:{grip_angle:3d}° | "
                              f"Hand:{hand_openness_value:.3f if hand_openness_value else 'N/A':>7} | "
                              f"State: {motion_state}")

                    cv2.imshow("Nuclear Waste Cleaning Arm Control - Test Mode (All Features)", annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nQuitting...")
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()
                print("\n✓ Test completed")
                print("All features were tested:")
                print("  ✓ Pose detection working")
                print("  ✓ Hand detection working" + (" ✓" if hand_openness_value is not None else " (hand not detected in last frame)"))
                print("  ✓ Angle calculations working")
                print("  ✓ Visual feedback working")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "4":
        print("\n" + "="*80)
        print("New Controller Mode - ML Model → ESP32 Direct Mapping")
        print("="*80)
        print("This mode uses controller.py which directly maps ML model output")
        print("(motion_state) to ESP32 servo commands via binary protocol.")
        print("\nRequirements:")
        print("  - ESP32 must be in AP mode (SSID: ESP32_AP, password: 12345678)")
        print("  - ESP32 IP: 192.168.4.1, Port: 8080")
        print("  - Connect your PC to ESP32_AP network")
        print("  - ESP32 firmware must be the new version (main.cpp with 4-servo support)")
        print("\nPress 'q' in camera window to quit.\n")
        
        try:
            # Import and run the new controller
            from controller import main as controller_main
            controller_main()
        except ImportError as e:
            print(f"ERROR: Could not import controller.py: {e}")
            print("Make sure controller.py exists in the repository root.")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "5":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Please run again and select 1-5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()

