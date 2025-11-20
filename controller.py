"""
Pose-to-Arm Controller
Maps detected fighting poses to 4-servo robotic arm movements via ESP32 TCP server.

================================================================================
COMPLETE DATA FLOW: ML MODEL → CONTROLLER → ESP32 → SERVOS
================================================================================

1. ML MODEL OUTPUT (ZonePoseDetector.process_frame):
   Input: Camera frame (numpy array)
   Output: tuple (annotated_frame, pose_type, angles, keypoints, arm_side, motion_state)
   
   motion_state dict structure:
   {
     "base": "LEFT" | "RIGHT" | "CENTER",      # From wrist horizontal position
     "forward": "FORWARD" | "BACKWARD" | "NEUTRAL",  # From shoulder_angle
     "vertical": "UP" | "DOWN" | "NEUTRAL",    # From elbow_angle
     "grip": "OPEN" | "CLOSE" | "HOLD"         # From wrist_angle
   }
   
   angles tuple: (shoulder_angle, elbow_angle, wrist_angle) in degrees

2. CONTROLLER MAPPING (map_pose_to_arm function):
   Input: motion_state dict from ML model
   Process: Maps motion_state values to servo angles using predefined rules
   Output: Calls ToyController.set_servo() for each servo (0-3)

3. ESP32 COMMUNICATION (ToyController.set_servo):
   Input: toy_id (0), servo_type (0-3), angle (0-180)
   Process: Packs to 3-byte binary: struct.pack('BBB', toy_id, servo_type, angle)
   Network: Sends via TCP to ESP32 at 192.168.4.1:8080
   Response: Waits for "OK" (2 bytes) acknowledgment

4. ESP32 FIRMWARE (main.cpp):
   Input: 3-byte TCP command (toy_id, servo_type, angle)
   Process: Validates servo_type < 4, calls setServoAngle(servo_type, angle)
   Hardware: Maps angle (0-180) to LEDC PWM duty cycle (minDuty-maxDuty)
   Output: Controls servo via LEDC channel on SERVO_PINS[servo_type]

5. SERVO MOVEMENT:
   Servo 0 (Base): Left/right rotation
   Servo 1 (Shoulder): Forward/backward motion
   Servo 2 (Elbow): Up/down motion
   Servo 3 (Claw): Open/close gripper

================================================================================
IMPORTANT NOTES:
================================================================================
- Run this script from the repository root so ml_model imports resolve correctly.
- If ML enum member names differ, map equivalents but preserve action semantics.
- If the ESP is switched to STA mode later, change ESP_HOST to the printed WiFi.localIP().
- Tune minDuty/maxDuty in firmware for your servos to avoid jitter.
- Tune the numeric angles below to match your mechanical geometry.
- Both controller.py and main.cpp MUST use the same servo_type indices (0-3).
"""

import sys
import os
import struct
import socket
import time
import cv2

# Ensure project root is in sys.path for ml_model imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_model.yolo_fightingpose_detection import ZonePoseDetector

# ESP32 TCP server configuration
ESP_HOST = "192.168.4.1"
ESP_PORT = 8080

# Default servo angles - TUNE THESE TO MATCH YOUR MECHANICAL GEOMETRY
neutral_base = 90
left_angle = 60
right_angle = 120

neutral_shoulder = 90
forward_angle = 120
back_angle = 60

neutral_elbow = 90
up_angle = 60
down_angle = 120

neutral_claw = 90
open_angle = 60
close_angle = 120

# Servo indices mapping (MUST MATCH firmware):
# servo_type = 0 → Base: left/right rotation (left = smaller angle, right = larger angle)
# servo_type = 1 → Shoulder: forward/backward motion (forward = larger angle, backward = smaller angle)
# servo_type = 2 → Elbow: up/down motion (up = smaller angle, down = larger angle)
# servo_type = 3 → Claw/Gripper: open/close (open = smaller angle, close = larger angle)


class ToyController:
    """Controller for robotic arm via ESP32 TCP server"""
    
    def __init__(self, host=ESP_HOST, port=ESP_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self):
        """Create TCP connection to ESP32"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2.0)  # 2 second timeout
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to ESP32 at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to ESP32: {e}")
            self.connected = False
            return False
    
    def set_servo(self, toy_id, servo_type, angle):
        """
        Send servo command to ESP32
        
        This function converts the ML model output (via map_pose_to_arm) into
        the 3-byte binary protocol expected by the ESP32 firmware.
        
        Protocol (matches main.cpp):
        - Byte 0: toy_id (currently 0)
        - Byte 1: servo_type (0-3: base, shoulder, elbow, claw)
        - Byte 2: angle (0-180 degrees)
        
        ESP32 responds with "OK" (2 bytes) on success.
        
        Args:
            toy_id: Toy identifier (keep as 0 for now)
            servo_type: Servo index (0-3: base, shoulder, elbow, claw)
            angle: Angle in degrees (0-180)
            
        Returns:
            bool: True if command acknowledged, False otherwise
        """
        if not self.connected or self.socket is None:
            return False
        
        # Validate inputs
        if servo_type < 0 or servo_type > 3:
            print(f"ERROR: Invalid servo_type {servo_type}, must be 0-3")
            return False
        
        angle = int(max(0, min(180, angle)))  # Clamp to valid range
        
        try:
            # Pack command as 3 bytes: toy_id, servo_type, angle
            # This binary format matches what ESP32 firmware expects in main.cpp
            command = struct.pack('BBB', toy_id, servo_type, angle)
            self.socket.sendall(command)
            
            # Wait for "OK" acknowledgment (2 bytes) - matches ESP32 firmware response
            response = self.socket.recv(2)
            if response == b"OK":
                return True
            else:
                print(f"ERROR: ESP32 unexpected response: {response} (expected b'OK')")
                return False
        except socket.timeout:
            print(f"ERROR: ESP32 command timeout (servo={servo_type}, angle={angle})")
            return False
        except Exception as e:
            print(f"ERROR: Failed to send ESP32 command: {e}")
            self.connected = False
            return False
    
    def close(self):
        """Close TCP socket"""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.connected = False
            print("Disconnected from ESP32")


def map_pose_to_arm(controller, motion_state, pose_type=None, angles=None):
    """
    Map ML model output (motion_state) to ESP32 servo commands
    
    ML MODEL OUTPUT STRUCTURE (from ZonePoseDetector.process_frame):
    - motion_state: dict with keys "base", "forward", "vertical", "grip"
      * "base": "LEFT" | "RIGHT" | "CENTER" (from wrist_x_norm position)
      * "forward": "FORWARD" | "BACKWARD" | "NEUTRAL" (from shoulder_angle)
      * "vertical": "UP" | "DOWN" | "NEUTRAL" (from elbow_angle)
      * "grip": "OPEN" | "CLOSE" | "HOLD" (from wrist_angle)
    - angles: tuple (shoulder_angle, elbow_angle, wrist_angle) in degrees
    - pose_type: PoseType enum (READY, RAISED, LOWERED, EXTENDED, UNKNOWN)
    
    ESP32 SERVO MAPPING (MUST MATCH firmware in main.cpp):
    - servo_type 0 → Base (left/right rotation)
    - servo_type 1 → Shoulder (forward/backward motion)
    - servo_type 2 → Elbow (up/down motion)
    - servo_type 3 → Claw/Gripper (open/close)
    
    This function implements the exact pose→arm mapping rules:
    - RIGHT_PUNCH/RIGHT_HOOK → base right then return
    - LEFT_PUNCH/LEFT_HOOK → base left then return
    - FORWARD_STANCE/ADVANCE → shoulder forward
    - BACK_STANCE/RETREAT → shoulder backward
    - WEAVE_UP/DUCK → elbow down
    - WEAVE_DOWN/UPLIFT → elbow up
    - GRAB/CLOSE_HAND → claw close
    - RELEASE/OPEN_HAND → claw open
    - GUARD → all servos to neutral
    """
    if not controller.connected:
        print("WARNING: Controller not connected, skipping arm movement")
        return
    
    if motion_state is None:
        print("WARNING: motion_state is None, cannot map to arm")
        return
    
    # Extract motion states from ML model output
    base_state = motion_state.get("base", "CENTER")
    forward_state = motion_state.get("forward", "NEUTRAL")
    vertical_state = motion_state.get("vertical", "NEUTRAL")
    grip_state = motion_state.get("grip", "HOLD")
    
    toy_id = 0  # Keep as 0 for now
    
    # Log ML output → ESP32 command mapping for debugging
    debug_msg = f"ML→ESP32: base={base_state}, forward={forward_state}, vertical={vertical_state}, grip={grip_state}"
    
    # Map base (servo 0): LEFT/RIGHT/CENTER
    if base_state == "RIGHT":
        # RIGHT_PUNCH/RIGHT_HOOK behavior: move right then return
        if controller.set_servo(toy_id, 0, right_angle):
            debug_msg += f" | Servo0→{right_angle}°"
        time.sleep(0.12)
        if controller.set_servo(toy_id, 0, neutral_base):
            debug_msg += f"→{neutral_base}°"
    elif base_state == "LEFT":
        # LEFT_PUNCH/LEFT_HOOK behavior: move left then return
        if controller.set_servo(toy_id, 0, left_angle):
            debug_msg += f" | Servo0→{left_angle}°"
        time.sleep(0.12)
        if controller.set_servo(toy_id, 0, neutral_base):
            debug_msg += f"→{neutral_base}°"
    elif base_state == "CENTER":
        # GUARD/neutral position for base
        if controller.set_servo(toy_id, 0, neutral_base):
            debug_msg += f" | Servo0={neutral_base}°"
    
    # Map forward/backward (servo 1: shoulder)
    if forward_state == "FORWARD":
        # FORWARD_STANCE/ADVANCE: shoulder forward
        if controller.set_servo(toy_id, 1, forward_angle):
            debug_msg += f" | Servo1→{forward_angle}°"
    elif forward_state == "BACKWARD":
        # BACK_STANCE/RETREAT: shoulder backward
        if controller.set_servo(toy_id, 1, back_angle):
            debug_msg += f" | Servo1→{back_angle}°"
    elif forward_state == "NEUTRAL":
        # GUARD/neutral position for shoulder
        if controller.set_servo(toy_id, 1, neutral_shoulder):
            debug_msg += f" | Servo1={neutral_shoulder}°"
    
    # Map vertical (servo 2: elbow)
    if vertical_state == "DOWN":
        # WEAVE_UP/DUCK: elbow down
        if controller.set_servo(toy_id, 2, down_angle):
            debug_msg += f" | Servo2→{down_angle}°"
    elif vertical_state == "UP":
        # WEAVE_DOWN/UPLIFT: elbow up
        if controller.set_servo(toy_id, 2, up_angle):
            debug_msg += f" | Servo2→{up_angle}°"
    elif vertical_state == "NEUTRAL":
        # GUARD/neutral position for elbow
        if controller.set_servo(toy_id, 2, neutral_elbow):
            debug_msg += f" | Servo2={neutral_elbow}°"
    
    # Map grip (servo 3: claw)
    if grip_state == "CLOSE":
        # GRAB/CLOSE_HAND: claw close
        if controller.set_servo(toy_id, 3, close_angle):
            debug_msg += f" | Servo3→{close_angle}°"
    elif grip_state == "OPEN":
        # RELEASE/OPEN_HAND: claw open
        if controller.set_servo(toy_id, 3, open_angle):
            debug_msg += f" | Servo3→{open_angle}°"
    elif grip_state == "HOLD":
        # GUARD/neutral position for claw
        if controller.set_servo(toy_id, 3, neutral_claw):
            debug_msg += f" | Servo3={neutral_claw}°"
    
    # Print debug message (can be disabled for production)
    # print(debug_msg)


def main():
    """Main control loop"""
    print("="*80)
    print(" " * 25 + "POSE-TO-ARM CONTROLLER")
    print("="*80)
    print("\nMake sure ESP32 is powered on and AP 'ESP32_AP' is active")
    print("Connect your PC to the ESP32_AP network (password: 12345678)")
    print("Press 'q' to quit\n")
    
    # Initialize pose detector
    print("Initializing pose detector...")
    detector = ZonePoseDetector()
    
    # Initialize controller
    controller = ToyController()
    if not controller.connect():
        print("Failed to connect to ESP32. Exiting.")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        controller.close()
        return
    
    print("Camera opened. Starting pose detection...")
    print("Move your arm to control the robotic arm. Press 'q' to quit.\n")
    print("="*80)
    print("ML MODEL → ESP32 CONNECTION ACTIVE")
    print("="*80)
    print("ML Output: motion_state dict with base/forward/vertical/grip")
    print("ESP32 Commands: 3-byte packets (toy_id, servo_type, angle)")
    print("="*80 + "\n")
    
    prev_motion_state = None
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame for pose detection - THIS IS THE ML MODEL OUTPUT
            annotated_frame, pose_type, angles, keypoints, arm_side, motion_state = detector.process_frame(frame)
            
            # Verify ML model output structure
            if motion_state is None:
                print(f"WARNING [Frame {frame_count}]: ML model returned None motion_state")
                continue
            
            # Validate motion_state has expected keys (from ML model)
            expected_keys = ["base", "forward", "vertical", "grip"]
            missing_keys = [key for key in expected_keys if key not in motion_state]
            if missing_keys:
                print(f"WARNING [Frame {frame_count}]: motion_state missing keys: {missing_keys}")
            
            # Only send commands if motion state changed (reduce jitter and ESP32 load)
            if motion_state != prev_motion_state:
                # LINK ML MODEL OUTPUT → ESP32 MOVEMENT
                # This is where the ML model output (motion_state) is converted to ESP32 servo commands
                map_pose_to_arm(controller, motion_state, pose_type, angles)
                prev_motion_state = motion_state.copy() if motion_state else None
                
                # Log the connection for first few frames
                if frame_count <= 5:
                    print(f"[Frame {frame_count}] ML→ESP32: {motion_state}")
            
            # Display annotated frame (shows ML model detection results)
            cv2.imshow("Pose Detection - Press 'q' to quit", annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        controller.close()
        print("Done.")


if __name__ == "__main__":
    main()

