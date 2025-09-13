import cv2
import mediapipe as mp
import numpy as np

class PersonObjectTracker:
    def __init__(self, chest_image_path=None):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Person state
        self.person_detected = False
        self.person_landmarks = None

        # Chest plate options
        self.use_image = chest_image_path is not None
        if self.use_image:
            # Load PNG with alpha channel
            chest_img = cv2.imread(chest_image_path, cv2.IMREAD_UNCHANGED)
            if chest_img is None:
                raise FileNotFoundError(f"Could not load chest image: {chest_image_path}")
            self.chest_image = chest_img
        else:
            self.chest_image = None

    def get_landmark_position(self, landmarks, landmark_type, frame_shape):
        """Convert landmark to pixel coords"""
        h, w = frame_shape[:2]
        lm = landmarks.landmark[landmark_type]
        if lm.visibility > 0.5:
            return int(lm.x * w), int(lm.y * h)
        return None

    def get_chest_center_and_dimensions(self, landmarks, frame_shape):
        """Calculate chest center point and dimensions based on torso landmarks"""
        h, w = frame_shape[:2]
        
        # Get key torso landmarks
        ls = self.get_landmark_position(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, frame_shape)
        rs = self.get_landmark_position(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, frame_shape)
        lh = self.get_landmark_position(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP, frame_shape)
        rh = self.get_landmark_position(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP, frame_shape)
        
        if not (ls and rs and lh and rh):
            return None, None, None
        
        # Calculate torso center
        shoulder_center = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        hip_center = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
        torso_center = ((shoulder_center[0] + hip_center[0]) // 2, 
                       (shoulder_center[1] + hip_center[1]) // 2)
        
        # Calculate dimensions
        shoulder_width = abs(rs[0] - ls[0])
        torso_height = abs(hip_center[1] - shoulder_center[1])
        
        return torso_center, shoulder_width, torso_height

    def draw_chest_polygon(self, frame, landmarks):
        """Draw chest plate as polygon anchored to shoulders/hips"""
        h, w = frame.shape[:2]

        ls = self.get_landmark_position(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, frame.shape)
        rs = self.get_landmark_position(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, frame.shape)
        lh = self.get_landmark_position(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP, frame.shape)
        rh = self.get_landmark_position(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP, frame.shape)

        if not (ls and rs and lh and rh):
            return

        # Create a more centered chest plate region
        # Calculate shoulder and hip centers for better alignment
        shoulder_center = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        hip_center = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
        
        # Create chest plate points that are better centered
        # Use a percentage of the torso width for consistent sizing
        torso_width = abs(rs[0] - ls[0])
        plate_width = int(torso_width * 0.8)  # 80% of shoulder width
        
        # Calculate vertical positioning
        shoulder_y = shoulder_center[1]
        hip_y = hip_center[1]
        plate_top = shoulder_y + int((hip_y - shoulder_y) * 0.1)  # Start 10% down from shoulders
        plate_bottom = shoulder_y + int((hip_y - shoulder_y) * 0.7)  # End 70% down towards hips
        
        # Center the plate horizontally
        center_x = (shoulder_center[0] + hip_center[0]) // 2
        plate_left = center_x - plate_width // 2
        plate_right = center_x + plate_width // 2
        
        # Create trapezoid shape (wider at top, narrower at bottom for natural chest shape)
        top_width = plate_width
        bottom_width = int(plate_width * 0.9)  # Slightly narrower at bottom
        
        top_left = (center_x - top_width // 2, plate_top)
        top_right = (center_x + top_width // 2, plate_top)
        bottom_right = (center_x + bottom_width // 2, plate_bottom)
        bottom_left = (center_x - bottom_width // 2, plate_bottom)
        
        pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

        if self.use_image:
            self.overlay_chest_image(frame, pts)
        else:
            cv2.fillPoly(frame, [pts], (0, 128, 255))   # orange plate
            cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
            cv2.putText(frame, "Chest Plate", (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def overlay_chest_image(self, frame, quad_pts):
        """Warp chest image onto trapezoid defined by body landmarks"""
        if self.chest_image is None:
            return

        # Source rectangle (image corners) - ensure correct order
        h, w = self.chest_image.shape[:2]
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Destination quadrilateral (ensure same order as source)
        dst_pts = np.float32(quad_pts)

        try:
            # Perspective transform
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Warp image
            warped = cv2.warpPerspective(self.chest_image, M, (frame.shape[1], frame.shape[0]))

            # Create mask for the chest plate area
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [quad_pts.astype(np.int32)], 255)

            # Blend using alpha channel if available
            if warped.shape[2] == 4:  # RGBA
                alpha = warped[:, :, 3] / 255.0
                alpha = alpha * (mask / 255.0)  # Apply mask to alpha
                
                for c in range(3):
                    frame[:, :, c] = (1 - alpha) * frame[:, :, c] + alpha * warped[:, :, c]
            else:
                # Use mask for blending
                warped_bgr = warped[:, :, :3] if warped.shape[2] > 3 else warped
                frame_masked = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
                warped_masked = cv2.bitwise_and(warped_bgr, warped_bgr, mask=mask)
                frame[:] = cv2.add(frame_masked, warped_masked)
                
        except cv2.error as e:
            print(f"Error in perspective transform: {e}")
            # Fallback to simple polygon
            cv2.fillPoly(frame, [quad_pts.astype(np.int32)], (0, 128, 255))

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            self.person_detected = True
            self.person_landmarks = results.pose_landmarks

            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

            # Draw chest plate
            self.draw_chest_polygon(frame, results.pose_landmarks)
        else:
            self.person_detected = False
            self.person_landmarks = None

        return frame


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = PersonObjectTracker(chest_image_path="kg82lbnu0ste1.png")  

    cv2.namedWindow("Chest Plate Tracker")

    print("Chest Plate Tracker")
    print("- Stand in front of camera to be detected")
    print("- Chest plate is anchored to shoulders + hips")
    print("- Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = tracker.process_frame(frame)

        if tracker.person_detected:
            cv2.putText(frame, "Person Detected!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No person detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Chest Plate Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()