import cv2
import numpy as np
import mediapipe as mp
import math
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class BodyPart:
    """Represents a segmented body part with its region and rotation point"""
    name: str
    region: Tuple[int, int, int, int]  # (x, y, width, height) in character image
    rotation_point: Tuple[float, float]  # Rotation center (normalized)
    parent: str = None  # Parent body part for hierarchical rotation

class AdvancedCharacterPoseMirror:
    def __init__(self, character_image_path):
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load character image
        self.character_img = cv2.imread(character_image_path, cv2.IMREAD_UNCHANGED)
        if self.character_img is None:
            raise ValueError(f"Could not load character image from {character_image_path}")
        
        # Convert to RGBA if needed
        if self.character_img.shape[2] == 3:
            self.character_img = cv2.cvtColor(self.character_img, cv2.COLOR_BGR2BGRA)
            self.character_img[:, :, 3] = 255
        
        self.char_height, self.char_width = self.character_img.shape[:2]
        
        # Define body parts for the business suit character
        # You'll need to adjust these coordinates based on your specific image
        self.body_parts = {
            'head': BodyPart(
                name='head',
                region=(int(self.char_width*0.35), -1500, int(self.char_width*0.3), int(self.char_height*0.3)),
                rotation_point=(0.5, 0.9)  # Neck area
            ),
            'torso': BodyPart(
                name='torso',
                region=(int(self.char_width*0.25), int(self.char_height*0.25), int(self.char_width*0.5), int(self.char_height*0.5)),
                rotation_point=(0.5, 0.1)  # Upper chest
            ),
            'right_upper_arm': BodyPart(
                name='right_upper_arm',
                region=(int(self.char_width*0.05), int(self.char_height*0.25), int(self.char_width*0.25), int(self.char_height*0.3)),
                rotation_point=(0.2, 0.2),  # Shoulder
                parent='torso'
            ),
            'left_upper_arm': BodyPart(
                name='left_upper_arm',
                region=(int(self.char_width*0.7), int(self.char_height*0.25), int(self.char_width*0.25), int(self.char_height*0.3)),
                rotation_point=(0.8, 0.2),  # Shoulder
                parent='torso'
            ),
            'left_forearm': BodyPart(
                name='left_forearm',
                region=(int(self.char_width*0.0), int(self.char_height*0.45), int(self.char_width*0.25), int(self.char_height*0.25)),
                rotation_point=(0.8, 0.1),  # Elbow
                parent='left_upper_arm'
            ),
            'right_forearm': BodyPart(
                name='right_forearm',
                region=(int(self.char_width*0.75), int(self.char_height*0.45), int(self.char_width*0.25), int(self.char_height*0.25)),
                rotation_point=(0.2, 0.1),  # Elbow
                parent='right_upper_arm'
            ),
            'legs': BodyPart(
                name='legs',
                region=(int(self.char_width*0.2), int(self.char_height*0.6), int(self.char_width*0.6), int(self.char_height*0.4)),
                rotation_point=(0.5, 0.1)  # Hip area
            )
        }
        
        # Extract body part images
        self.body_part_images = self.extract_body_parts()
        
        # Character reference pose landmarks (normalized coordinates)
        self.char_reference_pose = {
            'left_shoulder': (0.25, 0.28),
            'right_shoulder': (0.75, 0.28),
            'left_elbow': (0.15, 0.48),
            'right_elbow': (0.85, 0.48),
            'left_wrist': (0.12, 0.65),
            'right_wrist': (0.88, 0.65),
            'nose': (0.5, 0.15),
            'left_hip': (0.4, 0.62),
            'right_hip': (0.6, 0.62)
        }
        
        # Smoothing for pose estimation
        self.pose_history = []
        self.smoothing_frames = 5
        
    def extract_body_parts(self) -> Dict[str, np.ndarray]:
        """Extract individual body part images"""
        parts = {}
        
        for part_name, part_info in self.body_parts.items():
            x, y, w, h = part_info.region
            # Ensure coordinates are within image bounds
            x = max(0, min(x, self.char_width - 1))
            y = max(0, min(y, self.char_height - 1))
            w = min(w, self.char_width - x)
            h = min(h, self.char_height - y)
            
            part_img = self.character_img[y:y+h, x:x+w].copy()
            parts[part_name] = part_img
            
        return parts
    
    def smooth_pose(self, current_landmarks):
        """Apply temporal smoothing to reduce jitter"""
        self.pose_history.append(current_landmarks)
        if len(self.pose_history) > self.smoothing_frames:
            self.pose_history.pop(0)
        
        # Average the landmarks over the history
        if len(self.pose_history) == 1:
            return current_landmarks
        
        smoothed = {}
        for key in current_landmarks.keys():
            x_sum = sum([pose[key][0] for pose in self.pose_history])
            y_sum = sum([pose[key][1] for pose in self.pose_history])
            smoothed[key] = (x_sum / len(self.pose_history), y_sum / len(self.pose_history))
        
        return smoothed
    
    def calculate_rotation_angle(self, p1, p2, reference_angle=0):
        """Calculate rotation angle between two points"""
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        return angle - reference_angle
    
    def rotate_image_around_point(self, image, angle, pivot_point):
        """Rotate image around a specific pivot point"""
        h, w = image.shape[:2]
        pivot_x, pivot_y = int(pivot_point[0] * w), int(pivot_point[1] * h)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((pivot_x, pivot_y), math.degrees(angle), 1.0)
        
        # Calculate new image size to fit rotated image
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        
        # Adjust rotation matrix for the new center
        rotation_matrix[0, 2] += (new_w / 2) - pivot_x
        rotation_matrix[1, 2] += (new_h / 2) - pivot_y
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        
        return rotated, rotation_matrix
    
    def get_pose_landmarks(self, person_landmarks, frame_shape):
        """Extract and normalize pose landmarks"""
        h, w = frame_shape[:2]
        landmarks = {}
        
        landmark_mapping = {
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP
        }
        
        for name, landmark_idx in landmark_mapping.items():
            landmark = person_landmarks.landmark[landmark_idx]
            landmarks[name] = (landmark.x * w, landmark.y * h)
        
        return self.smooth_pose(landmarks)
    
    def calculate_body_part_transforms(self, person_landmarks):
        """Calculate transformation for each body part"""
        transforms = {}
        
        # Calculate scale based on shoulder width
        shoulder_width = abs(person_landmarks['right_shoulder'][0] - person_landmarks['left_shoulder'][0])
        char_shoulder_width = abs(self.char_reference_pose['right_shoulder'][0] - self.char_reference_pose['left_shoulder'][0]) * self.char_width
        scale = shoulder_width * 1.75 / char_shoulder_width * 0.8  # Adjust scale factor
        
        # Calculate body center
        body_center = (
            (person_landmarks['left_shoulder'][0] + person_landmarks['right_shoulder'][0]) / 2,
            (person_landmarks['left_shoulder'][1] + person_landmarks['right_shoulder'][1]) / 2
        )
        
        # Reference angles for neutral pose
        ref_left_upper_arm_angle = math.atan2(
            self.char_reference_pose['left_elbow'][1] - self.char_reference_pose['left_shoulder'][1],
            self.char_reference_pose['left_elbow'][0] - self.char_reference_pose['left_shoulder'][0]
        )
        ref_right_upper_arm_angle = math.atan2(
            self.char_reference_pose['right_elbow'][1] - self.char_reference_pose['right_shoulder'][1],
            self.char_reference_pose['right_elbow'][0] - self.char_reference_pose['right_shoulder'][0]
        )
        
        # Calculate angles for arms
        left_upper_arm_angle = -self.calculate_rotation_angle(
            person_landmarks['left_shoulder'], person_landmarks['left_elbow'], ref_left_upper_arm_angle
        )
        right_upper_arm_angle = -self.calculate_rotation_angle(
            person_landmarks['right_shoulder'], person_landmarks['right_elbow'], ref_right_upper_arm_angle
        )
        
        left_forearm_angle = -self.calculate_rotation_angle(
            person_landmarks['left_elbow'], person_landmarks['left_wrist']
        ) - left_upper_arm_angle
        
        right_forearm_angle = -self.calculate_rotation_angle(
            person_landmarks['right_elbow'], person_landmarks['right_wrist']
        ) - right_upper_arm_angle
        
        # Store transforms
        transforms['left_upper_arm'] = {
            'rotation': left_upper_arm_angle,
            'scale': scale,
            'position': person_landmarks['left_shoulder']
        }
        transforms['right_upper_arm'] = {
            'rotation': right_upper_arm_angle,
            'scale': scale,
            'position': person_landmarks['right_shoulder']
        }
        transforms['left_forearm'] = {
            'rotation': left_forearm_angle,
            'scale': scale,
            'position': person_landmarks['left_elbow']
        }
        transforms['right_forearm'] = {
            'rotation': right_forearm_angle,
            'scale': scale,
            'position': person_landmarks['right_elbow']
        }
        transforms['torso'] = {
            'rotation': 0,
            'scale': scale,
            'position': body_center
        }
        nose_x, nose_y = person_landmarks['nose']

        transforms['head'] = {
            'rotation': 0,
            'scale': scale,
            'position': (nose_x, nose_y + 50)  # shift down
        }
        transforms['legs'] = {
            'rotation': 0,
            'scale': scale,
            'position': ((person_landmarks['left_hip'][0] + person_landmarks['right_hip'][0]) / 2,
                        (person_landmarks['left_hip'][1] + person_landmarks['right_hip'][1]) / 2)
        }
        
        return transforms
    
    def render_character(self, frame, person_landmarks):
        """Render the character with transformed body parts"""
        if person_landmarks is None:
            return frame
        
        landmarks = self.get_pose_landmarks(person_landmarks, frame.shape)
        transforms = self.calculate_body_part_transforms(landmarks)
        
        # Create output frame
        result = frame.copy()
        
        # Render order (back to front)
        render_order = ['legs', 'torso', 'left_upper_arm', 'right_upper_arm', 
                       'left_forearm', 'right_forearm', 'head']
        
        for part_name in render_order:
            if part_name not in self.body_part_images or part_name not in transforms:
                continue
                
            part_img = self.body_part_images[part_name]
            part_info = self.body_parts[part_name]
            transform = transforms[part_name]
            
            # Apply rotation
            rotated_img, rotation_matrix = self.rotate_image_around_point(
                part_img, transform['rotation'], part_info.rotation_point
            )
            
            # Apply scaling
            new_size = (int(rotated_img.shape[1] * transform['scale']),
                       int(rotated_img.shape[0] * transform['scale']))
            if new_size[0] > 0 and new_size[1] > 0:
                scaled_img = cv2.resize(rotated_img, new_size, interpolation=cv2.INTER_LINEAR)
            else:
                continue
            
            # Calculate position
            pos_x = int(transform['position'][0] - scaled_img.shape[1] // 2)
            pos_y = int(transform['position'][1] - scaled_img.shape[0] // 2)
            
            # Blend with result
            result = self.blend_image(result, scaled_img, (pos_x, pos_y))
        
        return result
    
    def blend_image(self, background, overlay, position):
        """Blend overlay image with background at specified position"""
        x, y = position
        h, w = overlay.shape[:2]
        
        # Calculate overlapping region
        bg_h, bg_w = background.shape[:2]
        
        # Clamp coordinates
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + w)
        y2 = min(bg_h, y + h)
        
        if x1 >= x2 or y1 >= y2:
            return background
        
        # Calculate overlay region
        ov_x1 = x1 - x
        ov_y1 = y1 - y
        ov_x2 = ov_x1 + (x2 - x1)
        ov_y2 = ov_y1 + (y2 - y1)
        
        if overlay.shape[2] == 4:  # Has alpha channel
            alpha = overlay[ov_y1:ov_y2, ov_x1:ov_x2, 3:4] / 255.0
            overlay_rgb = overlay[ov_y1:ov_y2, ov_x1:ov_x2, :3]
            background_region = background[y1:y2, x1:x2]
            
            blended = background_region * (1 - alpha) + overlay_rgb * alpha
            background[y1:y2, x1:x2] = blended.astype(np.uint8)
        else:
            background[y1:y2, x1:x2] = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
        
        return background
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Advanced Character Pose Mirror")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'd' to toggle pose landmarks debug view")
        
        show_debug = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process pose
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Render character
                frame = self.render_character(frame, results.pose_landmarks)
                
                # Debug view
                if show_debug:
                    self.mp_draw.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                    )
            
            cv2.imshow('Advanced Character Mirror', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('character_mirror_advanced.jpg', frame)
                print("Frame saved!")
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    character_image_path = r"C:\Users\amrut\OneDrive\Desktop\Programs\HTN\Directors-Lens\qazicontentstuff\Human_body_outline.png"  # Replace with your image path
    
    try:
        app = AdvancedCharacterPoseMirror(character_image_path)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Your character image file")
        print("2. Required packages: pip install opencv-python mediapipe numpy")
        print("3. A working camera")
        print("\nNote: You may need to adjust the body part regions in the code")
        print("based on your specific character image anatomy.")

if __name__ == "__main__":
    main()