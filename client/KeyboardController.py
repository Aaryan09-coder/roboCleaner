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
        
        # Initial servo positions (match hardware axes)
        self.base_pos = 90      # Servo1 - Base left/right
        self.forward_pos = 90   # Servo2 - Forward/backward
        self.updown_pos = 90    # Servo3 - Up/down
        self.grip_pos = 90      # Servo4 - Grip
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
    
    def send_angles(self, servo1, servo2, servo3, servo4):
        """
        Send calculated angles to ESP32
        
        Args:
            servo1: Servo 1 angle - Base left/right (0-180)
            servo2: Servo 2 angle - Forward/backward (0-180)
            servo3: Servo 3 angle - Up/down (0-180)
            servo4: Servo 4 angle - Grip (0-180)
        """
        if not self.connected:
            return
        
        # Placeholder for actual communication with ESP32
        print(f"Servo1(base L/R): {servo1}° | Servo2(fwd/back): {servo2}° | Servo3(up/down): {servo3}° | Servo4(grip): {servo4}°")
    
    def run_keyboard_mode(self):
        """Run keyboard control mode"""
        print("Keyboard control mode started!")
        print("Controls:")
        print("  A/D - Servo1: Base left/right")
        print("  W/S - Servo2: Forward/backward")
        print("  Q/E - Servo3: Up/down")
        print("  Z/X - Servo4: Grip")
        print("  R   - Reset arm")
        print("  ESC - Exit")
        
        try:
            # Note: keyboard module needs admin privileges on Windows
            import keyboard
            
            while True:
                if keyboard.is_pressed('esc'):
                    break
                elif keyboard.is_pressed('a'):
                    self.base_pos = max(0, self.base_pos - self.step)
                    self.send_angles(self.base_pos, self.forward_pos, self.updown_pos, self.grip_pos)
                elif keyboard.is_pressed('d'):
                    self.base_pos = min(180, self.base_pos + self.step)
                    self.send_angles(self.base_pos, self.forward_pos, self.updown_pos, self.grip_pos)
                elif keyboard.is_pressed('w'):
                    self.forward_pos = min(180, self.forward_pos + self.step)
                    self.send_angles(self.base_pos, self.forward_pos, self.updown_pos, self.grip_pos)
                elif keyboard.is_pressed('s'):
                    self.forward_pos = max(0, self.forward_pos - self.step)
                    self.send_angles(self.base_pos, self.forward_pos, self.updown_pos, self.grip_pos)
                elif keyboard.is_pressed('q'):
                    self.updown_pos = max(0, self.updown_pos - self.step)
                    self.send_angles(self.base_pos, self.forward_pos, self.updown_pos, self.grip_pos)
                elif keyboard.is_pressed('e'):
                    self.updown_pos = min(180, self.updown_pos + self.step)
                    self.send_angles(self.base_pos, self.forward_pos, self.updown_pos, self.grip_pos)
                elif keyboard.is_pressed('z'):
                    self.grip_pos = max(0, self.grip_pos - self.step)
                    self.send_angles(self.base_pos, self.forward_pos, self.updown_pos, self.grip_pos)
                elif keyboard.is_pressed('x'):
                    self.grip_pos = min(180, self.grip_pos + self.step)
                    self.send_angles(self.base_pos, self.forward_pos, self.updown_pos, self.grip_pos)
                elif keyboard.is_pressed('r'):
                    self.base_pos = 90
                    self.forward_pos = 90
                    self.updown_pos = 90
                    self.grip_pos = 90
                    self.send_angles(self.base_pos, self.forward_pos, self.updown_pos, self.grip_pos)
                    print("Arm reset to neutral position")
        except ImportError:
            print("ERROR: keyboard module not available")
            print("Running in simulated mode...")
            print("(Real keyboard control requires admin privileges on Windows)")
