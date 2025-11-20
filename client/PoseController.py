"""
Pose-based controller for the robotic arm
Uses pose detection to drive servo motors via ESP32
Supports two control modes:
  1. Mode-switching (keyboard selects axis, pose controls it)
  2. Two-handed control (left arm = base+claw, right arm = forward+vertical)
"""
import cv2
import socket
import json
import numpy as np
from ml_model.yolo_fightingpose_detection import ZonePoseDetector

# Optional dependency for hand (finger) detection used for grip state
try:
    import importlib
    mp = importlib.import_module("mediapipe")
    _HAS_MEDIAPIPE = True
except Exception:
    mp = None
    _HAS_MEDIAPIPE = False


def show_control_mode_menu():
    """
    Display menu to choose control mode and show brief descriptions
    
    Returns:
        str: '1' for mode-switching, '2' for two-handed, 'q' to quit
    """
    print("\n" + "="*80)
    print(" " * 25 + "ROBOTIC ARM CONTROL MODES")
    print("="*80)
    print("\nChoose a control mode:\n")
    
    print("  [1] MODE-SWITCHING CONTROL (Recommended for Beginners)")
    print("  " + "-"*76)
    print("     • Use keyboard keys (1, 2, 3, 4) to select which axis to control")
    print("     • Then use your arm movements to control that specific axis")
    print("     • Perfect for precise, one-axis-at-a-time control")
    print("\n     Control Mapping:")
    print("       Key '1' → Base Left/Right (Servo1)")
    print("              Movement: Move wrist horizontally across screen")
    print("              Left side = 0°, Center = 90°, Right side = 180°")
    print("\n       Key '2' → Forward/Backward (Servo2)")
    print("              Movement: Raise/lower shoulder angle")
    print("              Backward (arm down) = 0°, Neutral = 90°, Forward (arm up) = 180°")
    print("\n       Key '3' → Up/Down (Servo3)")
    print("              Movement: Bend/unbend elbow")
    print("              Down (elbow bent) = 0°, Neutral = 90°, Up (arm straight) = 180°")
    print("\n       Key '4' → Claw Open/Close (Servo4)")
    print("              Movement: Open/close your hand")
    print("              Open hand = 0° (claw opens), Closed fist = 180° (claw closes)")
    
    print("\n  [2] TWO-HANDED CONTROL (Advanced)")
    print("  " + "-"*76)
    print("     • Use both arms simultaneously to control multiple axes")
    print("     • More natural and intuitive for experienced users")
    print("     • Control multiple axes at the same time")
    print("\n     Control Mapping:")
    print("       LEFT ARM:")
    print("         • Wrist X position → Base Left/Right (Servo1)")
    print("           Move left arm horizontally: Left = 0°, Right = 180°")
    print("         • Hand open/close → Claw Open/Close (Servo4)")
    print("           Open left hand = claw opens (0°), Close left hand = claw closes (180°)")
    print("\n       RIGHT ARM:")
    print("         • Shoulder angle → Forward/Backward (Servo2)")
    print("           Raise/lower right shoulder: Backward = 0°, Forward = 180°")
    print("         • Elbow angle → Up/Down (Servo3)")
    print("           Bend right elbow: Down = 0°, Up = 180°")
    
    print("\n  [Q] Quit")
    print("\n" + "="*80)
    
    choice = input("\nEnter your choice (1, 2, or Q): ").strip().lower()
    return choice


class PoseController:
    """Controls robotic arm using pose detection"""
    
    def __init__(self, esp32_host="192.168.1.100", esp32_port=8000):
        """
        Initialize pose controller
        
        Args:
            esp32_host: IP address of ESP32
            esp32_port: Port number for ESP32 communication
        """
        self.esp32_host = esp32_host
        self.esp32_port = esp32_port
        self.detector = ZonePoseDetector()
        self.connected = False
        self.socket = None

        # Optional MediaPipe hands detector (used for grip estimation)
        self.hands_detector = None
        self.left_hands_detector = None
        self.right_hands_detector = None
        
        if _HAS_MEDIAPIPE:
            try:
                solutions = getattr(mp, "solutions", None)
                hands_module = getattr(solutions, "hands", None) if solutions is not None else None
                if hands_module is not None:
                    # Single hand detector for mode-switching (support 2 hands for better detection)
                    self.hands_detector = hands_module.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,  # Lower threshold for better detection
                        min_tracking_confidence=0.3,
                        model_complexity=1,
                    )
                    # Two hand detectors for two-handed mode (support 2 hands)
                    self.left_hands_detector = hands_module.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,  # Lower threshold for better detection
                        min_tracking_confidence=0.3,
                        model_complexity=1,
                    )
                    self.right_hands_detector = hands_module.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,  # Lower threshold for better detection
                        min_tracking_confidence=0.3,
                        model_complexity=1,
                    )
            except Exception:
                pass
    
    def connect(self):
        """
        Connect to ESP32 via WiFi socket
        
        Returns:
            bool: True if connected, False otherwise
        """
        try:
            print(f"Attempting to connect to ESP32 at {self.esp32_host}:{self.esp32_port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)  # 5 second timeout
            self.socket.connect((self.esp32_host, self.esp32_port))
            self.connected = True
            print("✓ Connected to ESP32")
            return True
        except socket.timeout:
            print(f"✗ Connection timeout: ESP32 not responding at {self.esp32_host}:{self.esp32_port}")
            self.connected = False
            return False
        except socket.error as e:
            print(f"✗ Connection failed: {e}")
            print(f"  Make sure ESP32 is powered and WiFi is enabled")
            self.connected = False
            return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            self.connected = False
            return False
    
    def send_angles(self, servo1, servo2, servo3, servo4):
        """
        Send calculated angles to ESP32 via WiFi socket
        
        Args:
            servo1: Servo 1 angle - Base left/right (0-180)
            servo2: Servo 2 angle - Forward/backward (0-180)
            servo3: Servo 3 angle - Up/down (0-180)
            servo4: Servo 4 angle - Grip (0-180)
        """
        if not self.connected or self.socket is None:
            return False

        try:
            servo1_angle = int(np.clip(servo1, 0, 180))
            servo2_angle = int(np.clip(servo2, 0, 180))
            servo3_angle = int(np.clip(servo3, 0, 180))
            servo4_angle = int(np.clip(servo4, 0, 180))

            command = {
                "type": "servo",
                "servo1": servo1_angle,
                "servo2": servo2_angle,
                "servo3": servo3_angle,
                "servo4": servo4_angle
            }

            message = json.dumps(command) + "\n"
            self.socket.sendall(message.encode("utf-8"))
            return True
        except socket.error as e:
            print(f"✗ Send failed: {e}")
            self.connected = False
            self.disconnect()
            return False
        except Exception as e:
            print(f"✗ Error sending command: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False
    
    def detect_hand_openness(self, hands_detector, frame, wrist_x, wrist_y, arm_side, h, w, annotated_frame=None):
        """
        Detect hand openness using MediaPipe with multiple fallback strategies
        
        Args:
            hands_detector: MediaPipe hands detector
            frame: Full camera frame
            wrist_x, wrist_y: Wrist position in full frame coordinates
            arm_side: 'left' or 'right'
            h, w: Frame height and width
            annotated_frame: Optional frame to draw hand landmarks on
            
        Returns:
            tuple: (hand_openness_value, hand_landmarks) or (None, None) if detection fails
                hand_openness_value: float (0-1, smaller = closed, larger = open)
                hand_landmarks: MediaPipe hand landmarks or None
        """
        if hands_detector is None:
            return None, None
            
        try:
            # Strategy 1: Crop around wrist (larger crop for better detection)
            crop_size = int(min(h, w) * 0.4)  # Increased from 0.25 to 0.4
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
                    # Use the first detected hand
                    lm = res.multi_hand_landmarks[0].landmark
                    hand_side = res.multi_handedness[0].classification[0].label  # 'Left' or 'Right'
                    
                    # Match hand side with arm side (left hand on left arm, etc.)
                    # Note: MediaPipe's 'Left' means left hand (from hand's perspective)
                    # Our arm_side is from body's perspective
                    if (arm_side == 'left' and hand_side == 'Left') or \
                       (arm_side == 'right' and hand_side == 'Right'):
                        # Calculate hand openness
                        tip_idxs = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
                        wrist_lm = lm[0]  # Wrist landmark
                        
                        dists = []
                        for idx in tip_idxs:
                            dx = lm[idx].x - wrist_lm.x
                            dy = lm[idx].y - wrist_lm.y
                            dists.append((dx * dx + dy * dy) ** 0.5)
                        
                        avg = float(sum(dists) / len(dists))
                        
                        # Draw hand landmarks on annotated frame if provided
                        if annotated_frame is not None and mp is not None:
                            try:
                                solutions = getattr(mp, "solutions", None)
                                drawing_utils = getattr(solutions, "drawing_utils", None) if solutions else None
                                hands_module = getattr(solutions, "hands", None) if solutions else None
                                
                                if drawing_utils and hands_module:
                                    # Draw on full frame (scale landmarks from crop to full frame)
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
            
            # Strategy 2: Try full frame detection as fallback
            try:
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res_full = hands_detector.process(rgb_full)
                
                if res_full and res_full.multi_hand_landmarks:
                    # Find hand closest to wrist position
                    best_hand = None
                    best_dist = float('inf')
                    
                    for hand_idx, hand_landmarks in enumerate(res_full.multi_hand_landmarks):
                        # Get wrist position in full frame
                        wrist_lm_full = hand_landmarks.landmark[0]
                        wrist_x_full = wrist_lm_full.x * w
                        wrist_y_full = wrist_lm_full.y * h
                        
                        # Calculate distance to YOLO wrist position
                        dist = ((wrist_x_full - wrist_x) ** 2 + (wrist_y_full - wrist_y) ** 2) ** 0.5
                        
                        if dist < best_dist:
                            best_dist = dist
                            best_hand = hand_landmarks
                    
                    if best_hand is not None and best_dist < min(h, w) * 0.3:  # Within reasonable distance
                        lm = best_hand.landmark
                        tip_idxs = [4, 8, 12, 16, 20]
                        wrist_lm = lm[0]
                        
                        dists = []
                        for idx in tip_idxs:
                            dx = lm[idx].x - wrist_lm.x
                            dy = lm[idx].y - wrist_lm.y
                            dists.append((dx * dx + dy * dy) ** 0.5)
                        
                        avg = float(sum(dists) / len(dists))
                        
                        # Draw hand landmarks on full frame if provided
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
                
        except Exception as e:
            # Silently fail - return None
            pass
        
        return None, None
    
    def run_mode_switching(self):
        """
        Option 1: Mode-switching control
        Use keyboard keys (1,2,3,4) to select axis, then control with pose
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return
        
        print("\n" + "="*80)
        print("MODE-SWITCHING CONTROL ACTIVE")
        print("="*80)
        print("\nKeyboard Controls:")
        print("  Press '1' → Control Base Left/Right (Servo1)")
        print("  Press '2' → Control Forward/Backward (Servo2)")
        print("  Press '3' → Control Up/Down (Servo3)")
        print("  Press '4' → Control Claw Open/Close (Servo4)")
        print("  Press 'Q' → Quit")
        print("\nStart controlling by pressing 1, 2, 3, or 4...\n")
        
        # Current active mode (1=Base, 2=Forward, 3=Up/Down, 4=Claw)
        active_mode = 2  # Start with forward/backward
        mode_names = {
            1: "BASE LEFT/RIGHT (Servo1)",
            2: "FORWARD/BACKWARD (Servo2)",
            3: "UP/DOWN (Servo3)",
            4: "CLAW OPEN/CLOSE (Servo4)"
        }
        
        # Neutral servo positions
        servo1_angle = 90
        servo2_angle = 90
        servo3_angle = 90
        servo4_angle = 90
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('1'):
                    active_mode = 1
                    print(f"→ Mode changed to: {mode_names[1]}")
                elif key == ord('2'):
                    active_mode = 2
                    print(f"→ Mode changed to: {mode_names[2]}")
                elif key == ord('3'):
                    active_mode = 3
                    print(f"→ Mode changed to: {mode_names[3]}")
                elif key == ord('4'):
                    active_mode = 4
                    print(f"→ Mode changed to: {mode_names[4]}")
                
                # Process pose detection
                annotated_frame, pose, angles, keypoints, arm_side, motion_state = \
                    self.detector.process_frame(frame)
                
                shoulder_angle, elbow_angle, wrist_angle = angles
                
                # Control based on active mode
                if active_mode == 1:  # Base Left/Right
                    if keypoints is not None and arm_side is not None:
                        LEFT_WRIST_IDX = 9
                        RIGHT_WRIST_IDX = 10
                        wrist_idx = LEFT_WRIST_IDX if arm_side == 'left' else RIGHT_WRIST_IDX
                        
                        if wrist_idx < len(keypoints) and keypoints[wrist_idx][2] > 0.2:
                            wrist_x = keypoints[wrist_idx][0]
                            # Map wrist X position to servo angle: left edge = 0°, right edge = 180°
                            servo1_angle = int(np.clip((wrist_x / w) * 180.0, 0, 180))
                
                elif active_mode == 2:  # Forward/Backward
                    # Map shoulder angle: lower angle = backward (0°), higher angle = forward (180°)
                    # Typical range: 60-150 degrees maps to 0-180 servo
                    normalized_angle = np.clip((shoulder_angle - 60) / 90.0, 0, 1)
                    servo2_angle = int(np.clip(normalized_angle * 180.0, 0, 180))
                
                elif active_mode == 3:  # Up/Down
                    # Map elbow angle: more bent (smaller) = down (0°), straight (larger) = up (180°)
                    # Typical range: 60-180 degrees maps to 0-180 servo
                    normalized_angle = np.clip((elbow_angle - 60) / 120.0, 0, 1)
                    servo3_angle = int(np.clip(normalized_angle * 180.0, 0, 180))
                
                elif active_mode == 4:  # Claw Open/Close
                    if keypoints is not None and arm_side is not None:
                        LEFT_WRIST_IDX = 9
                        RIGHT_WRIST_IDX = 10
                        wrist_idx = LEFT_WRIST_IDX if arm_side == 'left' else RIGHT_WRIST_IDX
                        
                        if wrist_idx < len(keypoints) and keypoints[wrist_idx][2] > 0.2:
                            wrist_x = keypoints[wrist_idx][0]
                            wrist_y = keypoints[wrist_idx][1]
                            
                            # Try MediaPipe hand detection first
                            hand_openness, hand_landmarks = self.detect_hand_openness(
                                self.hands_detector, frame, wrist_x, wrist_y, arm_side, h, w, annotated_frame
                            )
                            
                            if hand_openness is not None:
                                # Map hand openness: open hand (smaller value) → 0°, closed (larger) → 180°
                                # Range: 0.30 (open) to 1.00 (closed)
                                servo4_angle = int(np.clip(np.interp(hand_openness, [0.30, 1.00], [0, 180]), 0, 180))
                                
                                # Display hand openness value
                                cv2.putText(annotated_frame, f"HandOpenness: {hand_openness:.3f}", 
                                           (10, h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            else:
                                # Fallback to wrist angle
                                servo4_angle = int(np.clip(wrist_angle, 0, 180))
                                cv2.putText(annotated_frame, "Hand: Not Detected (using wrist)", 
                                           (10, h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw visual feedback
                # Mode indicator
                mode_color = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)][active_mode - 1]
                cv2.rectangle(annotated_frame, (10, 10), (w - 10, 80), mode_color, 3)
                cv2.putText(annotated_frame, f"ACTIVE MODE: {mode_names[active_mode]}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
                cv2.putText(annotated_frame, "Press 1,2,3,4 to switch modes | Q to quit", 
                           (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Current servo angles
                y_offset = h - 100
                cv2.putText(annotated_frame, f"S1 (Base L/R): {servo1_angle}°", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 255) if active_mode == 1 else (200, 200, 200), 2)
                cv2.putText(annotated_frame, f"S2 (Forward/Back): {servo2_angle}°", 
                           (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (255, 0, 255) if active_mode == 2 else (200, 200, 200), 2)
                cv2.putText(annotated_frame, f"S3 (Up/Down): {servo3_angle}°", 
                           (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (255, 255, 0) if active_mode == 3 else (200, 200, 200), 2)
                cv2.putText(annotated_frame, f"S4 (Claw): {servo4_angle}°", 
                           (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 255, 0) if active_mode == 4 else (200, 200, 200), 2)
                
                # Movement instructions based on mode
                instructions = {
                    1: "Move your wrist horizontally: Left → Right",
                    2: "Raise/lower your shoulder: Backward → Forward",
                    3: "Bend/unbend elbow: Down → Up",
                    4: "Open/close your hand: Open → Close"
                }
                cv2.putText(annotated_frame, instructions[active_mode], 
                           (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Send angles to ESP32
                if self.connected:
                    self.send_angles(servo1_angle, servo2_angle, servo3_angle, servo4_angle)
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"[{frame_count}] Mode={active_mode} | "
                              f"S1:{servo1_angle}° S2:{servo2_angle}° "
                              f"S3:{servo3_angle}° S4:{servo4_angle}°")
                
                cv2.imshow("Mode-Switching Control", annotated_frame)
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.disconnect()
            print("\nMode-switching control ended")
    
    def run_two_handed(self):
        """
        Option 2: Two-handed control
        Left arm controls Base (L/R) and Claw, Right arm controls Forward/Back and Up/Down
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return
        
        print("\n" + "="*80)
        print("TWO-HANDED CONTROL ACTIVE")
        print("="*80)
        print("\nControl Instructions:")
        print("  LEFT ARM:")
        print("    • Move horizontally → Base Left/Right (Servo1)")
        print("    • Open/close hand → Claw Open/Close (Servo4)")
        print("  RIGHT ARM:")
        print("    • Raise/lower shoulder → Forward/Backward (Servo2)")
        print("    • Bend/unbend elbow → Up/Down (Servo3)")
        print("\nPress 'Q' to quit\n")
        
        # Neutral positions
        servo1_angle = 90  # Base
        servo2_angle = 90  # Forward/Back
        servo3_angle = 90  # Up/Down
        servo4_angle = 90  # Claw
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                
                # Process full frame for pose detection
                annotated_frame, pose, angles, keypoints, arm_side, motion_state = \
                    self.detector.process_frame(frame)
                
                # We need both arms, so detect both left and right separately
                # Run YOLO detection on full frame
                if self.detector.model is not None:
                    results = self.detector.model(frame, verbose=False)
                    
                    left_keypoints = None
                    right_keypoints = None
                    left_arm_side = 'left'
                    right_arm_side = 'right'
                    
                    if results and len(results) > 0:
                        result = results[0]
                        if result.keypoints is not None and len(result.keypoints) > 0:
                            all_keypoints = result.keypoints.data[0].cpu().numpy()
                            
                            # Check confidence for both arms
                            left_shoulder_conf = all_keypoints[self.detector.LEFT_SHOULDER][2] if \
                                len(all_keypoints) > self.detector.LEFT_SHOULDER else 0
                            right_shoulder_conf = all_keypoints[self.detector.RIGHT_SHOULDER][2] if \
                                len(all_keypoints) > self.detector.RIGHT_SHOULDER else 0
                            
                            # Extract keypoints for both arms (we have both in the same detection)
                            if left_shoulder_conf > 0.3:
                                left_keypoints = all_keypoints
                            
                            if right_shoulder_conf > 0.3:
                                right_keypoints = all_keypoints
                    
                    # Process LEFT ARM: Base (L/R) + Claw
                    if left_keypoints is not None:
                        left_wrist_idx = self.detector.LEFT_WRIST
                        if left_wrist_idx < len(left_keypoints) and \
                           left_keypoints[left_wrist_idx][2] > 0.3:
                            
                            left_wrist_x = left_keypoints[left_wrist_idx][0]
                            left_wrist_y = left_keypoints[left_wrist_idx][1]
                            
                            # Base Left/Right from left wrist X position
                            servo1_angle = int(np.clip((left_wrist_x / w) * 180.0, 0, 180))
                            
                            # Claw from left hand openness
                            left_hand_openness, left_hand_landmarks = self.detect_hand_openness(
                                self.left_hands_detector, frame, 
                                left_wrist_x, left_wrist_y, 'left', h, w, annotated_frame
                            )
                            
                            if left_hand_openness is not None:
                                servo4_angle = int(np.clip(
                                    np.interp(left_hand_openness, [0.30, 1.00], [0, 180]), 0, 180
                                ))
                            else:
                                # Keep previous value if detection fails
                                pass
                    
                    # Process RIGHT ARM: Forward/Back + Up/Down
                    if right_keypoints is not None:
                        right_shoulder_idx = self.detector.RIGHT_SHOULDER
                        right_elbow_idx = self.detector.RIGHT_ELBOW
                        right_wrist_idx = self.detector.RIGHT_WRIST
                        right_hip_idx = self.detector.RIGHT_HIP
                        
                        # Calculate angles for right arm
                        if (right_shoulder_idx < len(right_keypoints) and
                            right_elbow_idx < len(right_keypoints) and
                            right_keypoints[right_shoulder_idx][2] > 0.3 and
                            right_keypoints[right_elbow_idx][2] > 0.3):
                            
                            shoulder = right_keypoints[right_shoulder_idx][:2]
                            elbow = right_keypoints[right_elbow_idx][:2]
                            
                            # Get hip for shoulder angle calculation
                            if right_hip_idx < len(right_keypoints) and \
                               right_keypoints[right_hip_idx][2] > 0.3:
                                hip = right_keypoints[right_hip_idx][:2]
                            else:
                                hip = np.array([shoulder[0], shoulder[1] + 120.0])
                            
                            # Shoulder angle (Forward/Backward)
                            right_shoulder_angle = self.detector.calculate_angle(hip, shoulder, elbow)
                            normalized_angle = np.clip((right_shoulder_angle - 60) / 90.0, 0, 1)
                            servo2_angle = int(normalized_angle * 180.0)
                            
                            # Elbow angle (Up/Down)
                            if right_wrist_idx < len(right_keypoints) and \
                               right_keypoints[right_wrist_idx][2] > 0.3:
                                wrist = right_keypoints[right_wrist_idx][:2]
                                right_elbow_angle = self.detector.calculate_angle(shoulder, elbow, wrist)
                                normalized_elbow = np.clip((right_elbow_angle - 60) / 120.0, 0, 1)
                                servo3_angle = int(normalized_elbow * 180.0)
                    
                    # Draw visual indicators
                    # Split screen indicator
                    cv2.line(annotated_frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
                    
                    # Left side label
                    cv2.putText(annotated_frame, "LEFT ARM", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(annotated_frame, "Base L/R + Claw", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
                    
                    # Right side label
                    cv2.putText(annotated_frame, "RIGHT ARM", (w // 2 + 10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(annotated_frame, "Forward/Back + Up/Down", (w // 2 + 10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
                    
                    # Servo angles display
                    y_offset = h - 100
                    cv2.putText(annotated_frame, f"S1 (Base L/R): {servo1_angle}°", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(annotated_frame, f"S2 (Forward/Back): {servo2_angle}°", 
                               (w // 2 + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    cv2.putText(annotated_frame, f"S4 (Claw): {servo4_angle}°", 
                               (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"S3 (Up/Down): {servo3_angle}°", 
                               (w // 2 + 10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Send angles to ESP32
                if self.connected:
                    self.send_angles(servo1_angle, servo2_angle, servo3_angle, servo4_angle)
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"[{frame_count}] "
                              f"S1:{servo1_angle}° S2:{servo2_angle}° "
                              f"S3:{servo3_angle}° S4:{servo4_angle}°")
                
                cv2.imshow("Two-Handed Control", annotated_frame)
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.disconnect()
            print("\nTwo-handed control ended")
    
    def run_yolo_mode(self):
        """Legacy method - now redirects to menu selection"""
        choice = show_control_mode_menu()
        
        if choice == '1':
            self.run_mode_switching()
        elif choice == '2':
            self.run_two_handed()
        elif choice == 'q':
            print("Exiting...")
            return
        else:
            print("Invalid choice. Please run again and select 1, 2, or Q.")
