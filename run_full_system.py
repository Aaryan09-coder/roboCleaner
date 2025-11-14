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
                        
                        annotated_frame, pose, angles = detector.process_frame(frame)
                        
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
        print("  W/S - Shoulder up/down")
        print("  A/D - Elbow left/right")
        print("  Q/E - Wrist rotate left/right")
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
                    annotated_frame, pose, angles = detector.process_frame(frame)
                    
                    # Print updates every 30 frames or when pose changes
                    if pose and pose != current_pose:
                        current_pose = pose
                        print(f"[Frame {frame_count}] Pose: {pose.value}")
                        print(f"  Shoulder: {angles[0]:.1f}° | Elbow: {angles[1]:.1f}° | Wrist: {angles[2]:.1f}°")
                    elif frame_count % 30 == 0:
                        print(f"[Frame {frame_count}] Angles - Shoulder: {angles[0]:.1f}° | Elbow: {angles[1]:.1f}° | Wrist: {angles[2]:.1f}°")
                    
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

