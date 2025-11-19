"""
Pose-based controller for the robotic arm
Uses pose detection to drive servo motors via ESP32
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

        # Optional MediaPipe hands detector (used for grip estimation); default to None
        self.hands_detector = None
        if _HAS_MEDIAPIPE:
            try:
                # Safely access mp.solutions.hands to satisfy static analyzers and avoid None attribute errors
                solutions = getattr(mp, "solutions", None)
                hands_module = getattr(solutions, "hands", None) if solutions is not None else None
                if hands_module is not None:
                    # Create a hands detector with reasonable defaults
                    self.hands_detector = hands_module.Hands(
                        static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7,  # Stricter detection for robustness
                        min_tracking_confidence=0.5,
                        model_complexity=1,  # More robust complex model
                    )
                else:
                    # mediapipe is present but does not expose solutions.hands
                    self.hands_detector = None
            except Exception:
                # If instantiation fails for any reason, keep hands_detector as None
                self.hands_detector = None
    
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
        """
        Disconnect from ESP32
        """
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False
    
    def run_yolo_mode(self):
        """Run pose detection and control arm"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return
        
        print("Starting pose detection mode...")
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, pose, angles, keypoints, arm_side, motion_state = self.detector.process_frame(frame)

                # Extract angles: shoulder, elbow, wrist
                shoulder_angle, elbow_angle, wrist_angle = angles

                # Servo mappings:
                #   servo1 (base left/right): horizontal wrist/shoulder position
                #   servo2 (forward/backward): shoulder pitch
                #   servo3 (up/down): elbow bend
                #   servo4 (grip): detected hand openness (fallback to wrist angle)
                servo2_angle = int(np.clip(shoulder_angle, 0, 180))
                servo3_angle = int(np.clip(elbow_angle, 0, 180))
                servo4_angle = int(np.clip(wrist_angle, 0, 180))  # Default grip mapping

                h, w, _ = frame.shape
                servo1_angle = 90  # neutral base
                if keypoints is not None and arm_side is not None and w > 0:
                    LEFT_WRIST_IDX = 9
                    RIGHT_WRIST_IDX = 10
                    LEFT_SHOULDER_IDX = self.detector.LEFT_SHOULDER
                    RIGHT_SHOULDER_IDX = self.detector.RIGHT_SHOULDER
                    wrist_idx = LEFT_WRIST_IDX if arm_side == 'left' else RIGHT_WRIST_IDX
                    shoulder_idx = LEFT_SHOULDER_IDX if arm_side == 'left' else RIGHT_SHOULDER_IDX

                    base_source_idx = None
                    if wrist_idx < len(keypoints) and keypoints[wrist_idx][2] > 0.2:
                        base_source_idx = wrist_idx
                    elif shoulder_idx < len(keypoints) and keypoints[shoulder_idx][2] > 0.2:
                        base_source_idx = shoulder_idx

                    if base_source_idx is not None:
                        base_x = float(np.clip(keypoints[base_source_idx][0], 0, w))
                        servo1_angle = int(np.clip((base_x / w) * 180.0, 0, 180))

                if self.hands_detector is not None and keypoints is not None and arm_side is not None:
                    try:
                        # Choose wrist keypoint index based on arm_side
                        LEFT_WRIST_IDX = 9
                        RIGHT_WRIST_IDX = 10
                        wrist_idx = LEFT_WRIST_IDX if arm_side == 'left' else RIGHT_WRIST_IDX
                        # keypoints are in pixel coords (x,y,conf)
                        if keypoints is not None and wrist_idx < len(keypoints) and keypoints[wrist_idx][2] > 0.2:
                            print(f"[DEBUG] Wrist detected: conf={keypoints[wrist_idx][2]:.2f}")
                            wrist_xy = keypoints[wrist_idx][:2].astype(int)
                            h, w, _ = frame.shape
                            cx, cy = int(wrist_xy[0]), int(wrist_xy[1])
                            # Smaller crop for close-range hands (e.g., <30cm); tighter focus improves detection
                            crop_size = int(min(h, w) * 0.25)
                            half = crop_size // 2
                            x1 = max(0, cx - half)
                            y1 = max(0, cy - half)
                            x2 = min(w, cx + half)
                            y2 = min(h, cy + half)
                            print(f"[DEBUG] Crop ROI: ({x1},{y1}) to ({x2},{y2}), size={crop_size}x{crop_size}")
                            # draw ROI for debugging
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (200, 200, 0), 2)
                            crop = frame[y1:y2, x1:x2]
                            if crop.size == 0:
                                raise ValueError("Empty hand crop")
                            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            res = self.hands_detector.process(rgb_crop)
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
                                # NOTE: avg is normalized to the crop. For this user's camera and distance:
                                # Closed fist (fingertips close together): avg ~0.85-1.0
                                # Open hand (fingertips spread): avg ~0.35-0.45
                                # Map avg to grip angle: closed -> 180, open -> 0
                                servo4_angle = int(np.clip(np.interp(avg, [0.30, 1.00], [180, 0]), 0, 180))
                                print(f"[DEBUG] GripRaw: {avg:.3f} -> GripAngle: {servo4_angle}°")
                                cv2.putText(annotated_frame, f"GripRaw: {avg:.3f}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                                # Optionally draw landmarks on the cropped region for visual debug
                                try:
                                    if mp is not None and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'drawing_utils'):
                                        mp_draw = mp.solutions.drawing_utils
                                        # draw on the small crop then place back onto annotated_frame
                                        debug_crop = crop.copy()
                                        mp_draw.draw_landmarks(debug_crop, res.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
                                        # overlay debug_crop on top-left corner of annotated_frame
                                        th, tw = debug_crop.shape[:2]
                                        annotated_frame[0:th, 0:tw] = cv2.addWeighted(annotated_frame[0:th, 0:tw], 0.5, debug_crop, 0.5, 0)
                                except Exception:
                                    pass
                            else:
                                print(f"[DEBUG] No hand landmarks detected in crop")
                        else:
                            print(f"[DEBUG] Wrist keypoint not available or low confidence")
                    except Exception as e:
                        # Hand detection failed for this frame; keep fallback servo4_angle
                        print(f"[DEBUG] Hand detection exception: {e}")
                        pass

                # Send angles to ESP32 every frame
                if self.connected:
                    self.send_angles(servo1_angle, servo2_angle, servo3_angle, servo4_angle)
                    frame_count += 1
                    # Print status every 30 frames
                    if frame_count % 30 == 0:
                        print(f"[{frame_count}] Servo1(base L/R): {servo1_angle:.1f}°, Servo2(fwd/back): {servo2_angle:.1f}°, Servo3(up/down): {servo3_angle:.1f}°, Servo4(grip): {servo4_angle:.1f}° | Motion: {motion_state}")
                
                cv2.imshow("Nuclear Waste Cleaning Arm Control", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.disconnect()
            print("Pose detection mode ended")
