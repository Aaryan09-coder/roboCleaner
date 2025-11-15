"""
Pose-based controller for the robotic arm
Uses pose detection to drive servo motors via ESP32
"""
import cv2
import socket
import json
from ml_model.yolo_fightingpose_detection import ZonePoseDetector


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
    
    def send_angles(self, shoulder, elbow, wrist):
        """
        Send calculated angles to ESP32 via WiFi socket
        
        Args:
            shoulder: Shoulder angle (0-180)
            elbow: Elbow angle (0-180)
            wrist: Wrist angle (0-180)
        """
        if not self.connected or self.socket is None:
            return False
        
        try:
            # Create JSON command packet
            command = {
                "type": "servo_angles",
                "shoulder": int(shoulder),
                "elbow": int(elbow),
                "wrist": int(wrist)
            }
            
            # Send as JSON string with newline delimiter
            message = json.dumps(command) + "\n"
            self.socket.sendall(message.encode('utf-8'))
            
            return True
        except socket.error as e:
            print(f"✗ Send failed: {e}")
            self.connected = False
            self.disconnect()
            return False
        except Exception as e:
            print(f"✗ Error sending angles: {e}")
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
                
                annotated_frame, pose, angles = self.detector.process_frame(frame)
                
                # Send angles to ESP32 every frame
                if self.connected:
                    self.send_angles(angles[0], angles[1], angles[2])
                    frame_count += 1
                    # Print status every 30 frames
                    if frame_count % 30 == 0:
                        print(f"[{frame_count}] Shoulder: {angles[0]:.1f}°, Elbow: {angles[1]:.1f}°, Wrist: {angles[2]:.1f}°")
                
                cv2.imshow("Nuclear Waste Cleaning Arm Control", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.disconnect()
            print("Pose detection mode ended")
