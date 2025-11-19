"""
YOLO Pose Detection Module
Real-time pose detection using YOLO for robotic arm control
"""
import cv2
import numpy as np
from ultralytics import YOLO
from enum import Enum
from copy import deepcopy
import os
import shutil

class PoseType(Enum):
    """Enum for different pose types detected"""
    READY = "READY"
    RAISED = "RAISED"
    LOWERED = "LOWERED"
    EXTENDED = "EXTENDED"
    UNKNOWN = "UNKNOWN"


class ZonePoseDetector:
    """
    Detects human arm pose and calculates angles for robotic arm control
    Uses YOLO pose estimation to track keypoints and compute angles
    """
    
    def __init__(self, model_path="model_assets/yolo11n-pose.pt"):
        """
        Initialize the pose detector
        
        Args:
            model_path: Path to YOLO pose model file
        """
        self.model_path = model_path
        self.model = None
        self.initialize_model()
        
        # Keypoint indices for COCO pose format
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        # Confidence threshold for using keypoints in angle calculations
        self.conf_thresh = 0.5

        # Simple temporal smoothing for angles (exponential moving average)
        self.prev_angles = None
        self.smoothing_alpha = 0.4  # 0 = no smoothing, 1 = immediate

        # Discrete motion intent tracking (prevents accidental activations)
        self.intent_confirmation_frames = 3
        self.motion_state = {
            "base": "CENTER",
            "forward": "NEUTRAL",
            "vertical": "NEUTRAL",
            "grip": "HOLD",
        }
        self.motion_trackers = {
            axis: {"candidate": state, "streak": self.intent_confirmation_frames}
            for axis, state in self.motion_state.items()
        }

        # Thresholds for classifying discrete poses
        self.base_left_thresh = 0.42
        self.base_right_thresh = 0.58
        self.forward_lower_thresh = 65.0   # degrees
        self.forward_upper_thresh = 120.0  # degrees
        self.vertical_up_thresh = 70.0     # degrees
        self.vertical_down_thresh = 135.0  # degrees
        self.grip_open_thresh = 55.0       # degrees
        self.grip_close_thresh = 125.0     # degrees
        
    def initialize_model(self):
        """Initialize YOLO pose model"""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading YOLO model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                print(f"Model not found at {self.model_path}, downloading (online fallback)...")
                # Try to load the common ultralytics pose file name; YOLO will download it to cwd
                fallback_name = "yolov8n-pose.pt"
                self.model = YOLO(fallback_name)
                # If the downloaded file exists in cwd, copy it to the expected model path so next run uses local file
                try:
                    if os.path.exists(fallback_name):
                        dest_dir = os.path.dirname(self.model_path) or "."
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.copy(fallback_name, self.model_path)
                        print(f"Downloaded model saved to {self.model_path}")
                    else:
                        print("Note: fallback model was loaded but the file wasn't found in the current directory to copy.")
                except Exception as copy_err:
                    print(f"Warning: failed to copy downloaded model to {self.model_path}: {copy_err}")
                print("YOLO model loaded (online fallback)")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def calculate_angle(self, p1, p2, p3):
        """
        Calculate angle between three points
        
        Args:
            p1, p2, p3: Points as (x, y) tuples. p2 is the vertex
            
        Returns:
            Angle in degrees
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Vectors from p2 to p1 and p2 to p3
        v1 = np.array([x1 - x2, y1 - y2])
        v2 = np.array([x3 - x2, y3 - y2])
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def get_arm_side(self, keypoints):
        """
        Determine which arm to track (left or right)
        Uses the arm that has more confident detections
        
        Args:
            keypoints: Array of keypoints with confidence scores
            
        Returns:
            'left' or 'right'
        """
        left_shoulder_conf = keypoints[self.LEFT_SHOULDER][2] if len(keypoints) > self.LEFT_SHOULDER else 0
        right_shoulder_conf = keypoints[self.RIGHT_SHOULDER][2] if len(keypoints) > self.RIGHT_SHOULDER else 0
        
        return 'left' if left_shoulder_conf >= right_shoulder_conf else 'right'
    
    def process_frame(self, frame):
        """
        Process a single frame for pose detection
        
        Args:
            frame: Input image frame (numpy array)
            
        Returns:
            tuple: (annotated_frame, pose_type, angles, keypoints, arm_side, motion_state)
                - annotated_frame: Frame with drawn keypoints and skeleton
                - pose_type: Detected PoseType enum
                - angles: [shoulder_angle, elbow_angle, wrist_angle] in degrees
                - keypoints: numpy array of keypoints (N x 3) or None
                - arm_side: 'left' or 'right' (which arm was used)
                - motion_state: dict describing discrete commands for each axis
        """
        try:
            # Run YOLO inference
            if self.model is None:
                motion_state = deepcopy(self.motion_state)
                self._annotate_motion_state(frame, motion_state)
                return frame, PoseType.UNKNOWN, (0, 0, 0), None, None, motion_state
            results = self.model(frame, verbose=False)
            annotated_frame = frame.copy()
            h, w = frame.shape[:2]
            wrist_x_norm = 0.5
            # placeholders in case no keypoints are found
            keypoints = None
            arm_side = None
            
            # Default angles and pose
            shoulder_angle = 0
            elbow_angle = 0
            wrist_angle = 0
            pose_type = PoseType.UNKNOWN
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.keypoints is not None and len(result.keypoints) > 0:
                    keypoints = result.keypoints.data[0].cpu().numpy()
                    
                    # Determine which arm to track
                    arm_side = self.get_arm_side(keypoints)
                    
                    if arm_side == 'left':
                        shoulder_idx = self.LEFT_SHOULDER
                        elbow_idx = self.LEFT_ELBOW
                        wrist_idx = self.LEFT_WRIST
                    else:
                        shoulder_idx = self.RIGHT_SHOULDER
                        elbow_idx = self.RIGHT_ELBOW
                        wrist_idx = self.RIGHT_WRIST
                    
                    # Get keypoint coordinates (x, y, confidence)
                    shoulder = keypoints[shoulder_idx][:2]
                    elbow = keypoints[elbow_idx][:2]
                    wrist = keypoints[wrist_idx][:2]
                    
                    # Check if all keypoints are detected (confidence > 0.5)
                    if (keypoints[shoulder_idx][2] > self.conf_thresh and 
                        keypoints[elbow_idx][2] > self.conf_thresh and 
                        keypoints[wrist_idx][2] > self.conf_thresh):
                        
                        # Track wrist/base horizontal position for base rotation intent
                        base_source_idx = None
                        if wrist_idx < len(keypoints) and keypoints[wrist_idx][2] > self.conf_thresh:
                            base_source_idx = wrist_idx
                        elif shoulder_idx < len(keypoints) and keypoints[shoulder_idx][2] > self.conf_thresh:
                            base_source_idx = shoulder_idx
                        if base_source_idx is not None and w > 0:
                            base_x = float(np.clip(keypoints[base_source_idx][0], 0, w))
                            wrist_x_norm = base_x / max(w, 1e-6)

                        # Calculate angles
                        hip_idx = self.LEFT_HIP if arm_side == 'left' else self.RIGHT_HIP
                        # Use detected hip if confident, otherwise estimate a point below the shoulder
                        hip_conf = keypoints[hip_idx][2] if hip_idx < len(keypoints) else 0
                        if hip_conf > self.conf_thresh:
                            hip = keypoints[hip_idx][:2]
                        else:
                            # estimate hip as fixed offset below shoulder (pixels)
                            hip = np.array([shoulder[0], shoulder[1] + 120.0])
                        
                        # Shoulder angle (between hip-shoulder-elbow)
                        shoulder_angle = self.calculate_angle(hip, shoulder, elbow)
                        
                        # Elbow angle (between shoulder-elbow-wrist)
                        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                        
                        # Wrist angle: use forearm orientation (elbow -> wrist)
                        # Convert to an angle in degrees relative to horizontal.
                        forearm = np.array(wrist) - np.array(elbow)
                        # image y axis points down, invert y for standard Cartesian angles
                        fx, fy = forearm[0], -forearm[1]
                        wrist_angle_rad = np.arctan2(fy, fx)
                        wrist_angle = abs(np.degrees(wrist_angle_rad))
                        
                        # Map angles to servo angles (0-180)
                        shoulder_servo = np.clip(shoulder_angle, 0, 180)
                        elbow_servo = np.clip(elbow_angle, 0, 180)
                        wrist_servo = np.clip(wrist_angle, 0, 180)

                        # Apply simple temporal smoothing to reduce jitter
                        if self.prev_angles is None:
                            smoothed = (shoulder_servo, elbow_servo, wrist_servo)
                        else:
                            alpha = self.smoothing_alpha
                            smoothed = (
                                alpha * shoulder_servo + (1 - alpha) * self.prev_angles[0],
                                alpha * elbow_servo + (1 - alpha) * self.prev_angles[1],
                                alpha * wrist_servo + (1 - alpha) * self.prev_angles[2],
                            )
                        self.prev_angles = smoothed
                        shoulder_servo, elbow_servo, wrist_servo = smoothed
                        
                        # Determine pose type based on angles
                        if shoulder_angle < 60:
                            pose_type = PoseType.LOWERED
                        elif shoulder_angle > 120:
                            pose_type = PoseType.RAISED
                        elif elbow_angle > 150:
                            pose_type = PoseType.EXTENDED
                        else:
                            pose_type = PoseType.READY
                    
                        # Draw skeleton
                        self._draw_skeleton(annotated_frame, keypoints, arm_side)
            
            # Add text with angle information
            cv2.putText(annotated_frame, f"Shoulder: {shoulder_angle:.1f}°", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Elbow: {elbow_angle:.1f}°", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Wrist: {wrist_angle:.1f}°", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Pose: {pose_type.value}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            motion_state = self._classify_motion(shoulder_angle, elbow_angle, wrist_angle, wrist_x_norm)
            self._annotate_motion_state(annotated_frame, motion_state)
            
            return (
                annotated_frame,
                pose_type,
                (shoulder_angle, elbow_angle, wrist_angle),
                (keypoints if 'keypoints' in locals() else None),
                (arm_side if 'arm_side' in locals() else None),
                deepcopy(motion_state),
            )
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            motion_state = deepcopy(self.motion_state)
            self._annotate_motion_state(frame, motion_state)
            return frame, PoseType.UNKNOWN, (0, 0, 0), None, None, motion_state
    
    def _classify_motion(self, shoulder_angle, elbow_angle, wrist_angle, wrist_x_norm):
        """
        Convert continuous angles into discrete motion states with hysteresis.
        This reduces accidental activations caused by single-frame noise.
        """
        candidate = {}

        if wrist_x_norm < self.base_left_thresh:
            candidate["base"] = "LEFT"
        elif wrist_x_norm > self.base_right_thresh:
            candidate["base"] = "RIGHT"
        else:
            candidate["base"] = "CENTER"

        if shoulder_angle <= self.forward_lower_thresh:
            candidate["forward"] = "BACKWARD"
        elif shoulder_angle >= self.forward_upper_thresh:
            candidate["forward"] = "FORWARD"
        else:
            candidate["forward"] = "NEUTRAL"

        if elbow_angle <= self.vertical_up_thresh:
            candidate["vertical"] = "UP"
        elif elbow_angle >= self.vertical_down_thresh:
            candidate["vertical"] = "DOWN"
        else:
            candidate["vertical"] = "NEUTRAL"

        if wrist_angle <= self.grip_open_thresh:
            candidate["grip"] = "OPEN"
        elif wrist_angle >= self.grip_close_thresh:
            candidate["grip"] = "CLOSE"
        else:
            candidate["grip"] = "HOLD"

        for axis, intent in candidate.items():
            tracker = self.motion_trackers[axis]
            if intent == tracker["candidate"]:
                tracker["streak"] = min(self.intent_confirmation_frames, tracker["streak"] + 1)
            else:
                tracker["candidate"] = intent
                tracker["streak"] = 1

            if tracker["streak"] >= self.intent_confirmation_frames:
                self.motion_state[axis] = tracker["candidate"]

        return self.motion_state

    def _annotate_motion_state(self, frame, motion_state):
        """Overlay discrete motion state on the frame."""
        if frame is None or motion_state is None:
            return
        cv2.putText(frame, f"Base: {motion_state['base']}", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, f"Forward/Back: {motion_state['forward']}", (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, f"Up/Down: {motion_state['vertical']}", (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, f"Grip: {motion_state['grip']}", (10, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    def _draw_skeleton(self, frame, keypoints, arm_side='right'):
        """
        Draw pose skeleton on frame
        
        Args:
            frame: Image to draw on
            keypoints: Detected keypoints
            arm_side: 'left' or 'right'
        """
        # Define skeleton connections
        if arm_side == 'left':
            connections = [
                (self.LEFT_SHOULDER, self.LEFT_ELBOW),
                (self.LEFT_ELBOW, self.LEFT_WRIST),
            ]
            points_to_draw = [self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST]
        else:
            connections = [
                (self.RIGHT_SHOULDER, self.RIGHT_ELBOW),
                (self.RIGHT_ELBOW, self.RIGHT_WRIST),
            ]
            points_to_draw = [self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:
                start = tuple(map(int, keypoints[start_idx][:2]))
                end = tuple(map(int, keypoints[end_idx][:2]))
                cv2.line(frame, start, end, (0, 255, 0), 2)
        
        # Draw keypoints
        for idx in points_to_draw:
            if keypoints[idx][2] > 0.5:
                pos = tuple(map(int, keypoints[idx][:2]))
                cv2.circle(frame, pos, 5, (0, 255, 0), -1)
    
    def reset_arm(self):
        """Reset arm to neutral position"""
        pass
