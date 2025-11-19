#!/usr/bin/env python
"""
Standalone Pose Detection Test
Tests pose detection and angle calculation WITHOUT ESP32 connection
Just run this to verify the system works on your laptop
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_model.yolo_fightingpose_detection import ZonePoseDetector, ArmPose
import cv2

def main():
    print("="*70)
    print("STANDALONE POSE DETECTION TEST")
    print("="*70)
    print("\nThis test runs pose detection locally without ESP32 connection.")
    print("It will show you:")
    print("  - Real-time pose detection")
    print("  - Calculated arm angles (Shoulder, Elbow, Wrist)")
    print("  - Visual feedback on camera feed")
    print("\n" + "="*70)
    
    # Initialize detector
    print("\n[1/4] Loading YOLO model...")
    try:
        detector = ZonePoseDetector()
        print("    ✓ Model loaded successfully")
    except Exception as e:
        print(f"    ✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Open camera
    print("\n[2/4] Opening camera...")
    try:
        # Try different camera indices
        cap = None
        for camera_idx in [0, 1, 2]:
            print(f"    Trying camera index {camera_idx}...")
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"    ✓ Camera {camera_idx} opened successfully")
                    break
                else:
                    cap.release()
                    cap = None
            else:
                if cap:
                    cap.release()
                cap = None
        
        if cap is None or not cap.isOpened():
            print("    ✗ ERROR: Could not open any camera")
            print("    Make sure your camera is connected and not used by another app")
            return
        
        # Set camera properties for better performance
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        except:
            pass  # Some cameras don't support these settings
        
    except Exception as e:
        print(f"    ✗ ERROR opening camera: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[3/4] Starting pose detection...")
    print("    ✓ Ready!")
    
    print("\n" + "="*70)
    print("POSE DETECTION ACTIVE")
    print("="*70)
    print("\nInstructions:")
    print("  - Stand in front of the camera")
    print("  - Move your right arm to see angle calculations")
    print("  - Angles are displayed on screen and in console")
    print("  - Press 'q' to quit\n")
    
    current_pose = None
    frame_count = 0
    
    try:
        while True:
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("WARNING: Failed to capture frame, retrying...")
                    continue
                
                frame_count += 1
                
                # Process frame
                try:
                    annotated_frame, pose, angles, *_ = detector.process_frame(frame)
                except Exception as e:
                    print(f"ERROR in process_frame: {e}")
                    continue
                
                # Print updates when pose changes or every 60 frames
                if pose and pose != current_pose:
                    current_pose = pose
                    print(f"[Frame {frame_count:4d}] Pose: {pose.value:20s} | "
                          f"Shoulder: {angles[0]:6.1f}° | "
                          f"Elbow: {angles[1]:6.1f}° | "
                          f"Wrist: {angles[2]:6.1f}°")
                elif frame_count % 60 == 0:  # Print every 60 frames (~2 seconds at 30fps)
                    print(f"[Frame {frame_count:4d}] Angles - "
                          f"Shoulder: {angles[0]:6.1f}° | "
                          f"Elbow: {angles[1]:6.1f}° | "
                          f"Wrist: {angles[2]:6.1f}°")
                
                # Display frame
                try:
                    if annotated_frame is not None:
                        cv2.imshow("Pose Detection Test - Press 'q' to quit", annotated_frame)
                    else:
                        cv2.imshow("Pose Detection Test - Press 'q' to quit", frame)
                except Exception as e:
                    print(f"ERROR displaying frame: {e}")
                    # Try to continue without display
                    pass
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r'):  # Reset angles display
                    print("\n[Reset] Resetting angle history...")
                    detector.angle_history = []
                    detector.pose_history = []
                    
            except Exception as e:
                print(f"ERROR in main loop: {e}")
                import traceback
                traceback.print_exc()
                # Continue trying
                continue
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\nERROR during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[4/4] Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("    ✓ Camera released")
        print("    ✓ Windows closed")
        print("\n" + "="*70)
        print("TEST COMPLETE")
        print("="*70)
        print("\nIf you saw pose detection working, the system is ready!")
        print("Next step: Connect ESP32 and run the full system.")

if __name__ == "__main__":
    main()

