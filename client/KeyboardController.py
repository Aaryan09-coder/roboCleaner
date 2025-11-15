"""
Keyboard-based controller for the robotic arm
Uses keyboard input to control servo motors via ESP32
"""
import sys


class KeyboardController:
    """Controls robotic arm using keyboard input"""
    
    def __init__(self, esp32_host="192.168.1.100", esp32_port=8000):
        """
        Initialize keyboard controller
        
        Args:
            esp32_host: IP address of ESP32
            esp32_port: Port number for ESP32 communication
        """
        self.esp32_host = esp32_host
        self.esp32_port = esp32_port
        self.connected = False
        
        # Initial servo positions
        self.shoulder_pos = 90
        self.elbow_pos = 90
        self.wrist_pos = 90
        self.step = 5  # Angle increment per key press
    
    def connect(self):
        """
        Connect to ESP32
        
        Returns:
            bool: True if connected, False otherwise
        """
        try:
            print(f"Attempting to connect to ESP32 at {self.esp32_host}:{self.esp32_port}...")
            # Placeholder for actual ESP32 connection logic
            self.connected = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            self.connected = False
            return False
    
    def send_angles(self, shoulder, elbow, wrist):
        """
        Send calculated angles to ESP32
        
        Args:
            shoulder: Shoulder angle (0-180)
            elbow: Elbow angle (0-180)
            wrist: Wrist angle (0-180)
        """
        if not self.connected:
            return
        
        # Placeholder for actual communication with ESP32
        print(f"Shoulder: {shoulder}° | Elbow: {elbow}° | Wrist: {wrist}°")
    
    def run_keyboard_mode(self):
        """Run keyboard control mode"""
        print("Keyboard control mode started!")
        print("Controls:")
        print("  W/S - Shoulder up/down")
        print("  A/D - Elbow left/right")
        print("  Q/E - Wrist rotate left/right")
        print("  R   - Reset arm")
        print("  ESC - Exit")
        
        try:
            # Note: keyboard module needs admin privileges on Windows
            import keyboard
            
            while True:
                if keyboard.is_pressed('esc'):
                    break
                elif keyboard.is_pressed('w'):
                    self.shoulder_pos = min(180, self.shoulder_pos + self.step)
                    self.send_angles(self.shoulder_pos, self.elbow_pos, self.wrist_pos)
                elif keyboard.is_pressed('s'):
                    self.shoulder_pos = max(0, self.shoulder_pos - self.step)
                    self.send_angles(self.shoulder_pos, self.elbow_pos, self.wrist_pos)
                elif keyboard.is_pressed('a'):
                    self.elbow_pos = max(0, self.elbow_pos - self.step)
                    self.send_angles(self.shoulder_pos, self.elbow_pos, self.wrist_pos)
                elif keyboard.is_pressed('d'):
                    self.elbow_pos = min(180, self.elbow_pos + self.step)
                    self.send_angles(self.shoulder_pos, self.elbow_pos, self.wrist_pos)
                elif keyboard.is_pressed('q'):
                    self.wrist_pos = max(0, self.wrist_pos - self.step)
                    self.send_angles(self.shoulder_pos, self.elbow_pos, self.wrist_pos)
                elif keyboard.is_pressed('e'):
                    self.wrist_pos = min(180, self.wrist_pos + self.step)
                    self.send_angles(self.shoulder_pos, self.elbow_pos, self.wrist_pos)
                elif keyboard.is_pressed('r'):
                    self.shoulder_pos = 90
                    self.elbow_pos = 90
                    self.wrist_pos = 90
                    self.send_angles(self.shoulder_pos, self.elbow_pos, self.wrist_pos)
                    print("Arm reset to neutral position")
        except ImportError:
            print("ERROR: keyboard module not available")
            print("Running in simulated mode...")
            print("(Real keyboard control requires admin privileges on Windows)")
