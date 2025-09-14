import os
import sys
import math
from dataclasses import dataclass
from typing import Tuple, Dict, List
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class BodyPart:
    """Represents a segmented body part with its region and rotation point"""
    name: str
    region: Tuple[int, int, int, int]  # (x, y, width, height) in character image
    rotation_point: Tuple[float, float]  # Rotation center (normalized)
    parent: str = None  # Parent body part for hierarchical rotation


class AdvancedCharacterPoseMirror:
    """
    Based on your vfilm.py: MediaPipe pose + layered character parts rendered on a live camera frame,
    now augmented with a clickable REC/STOP button and VideoWriter.
    """
    def __init__(self, character_image_path: str):
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

        # Load character image (expects RGBA for alpha blending)
        self.character_img = cv2.imread(character_image_path, cv2.IMREAD_UNCHANGED)
        if self.character_img is None:
            raise ValueError(f"Could not load character image from {character_image_path}")

        if self.character_img.ndim == 2:
            self.character_img = cv2.cvtColor(self.character_img, cv2.COLOR_GRAY2BGRA)
        elif self.character_img.shape[2] == 3:
            self.character_img = cv2.cvtColor(self.character_img, cv2.COLOR_BGR2BGRA)
            self.character_img[:, :, 3] = 255

        self.char_height, self.char_width = self.character_img.shape[:2]

        # Define body parts (adjust these regions for your artwork)
        self.body_parts = {
            'head': BodyPart(
                name='head',
                region=(int(self.char_width * 0.35), int(self.char_height * 0.00), int(self.char_width * 0.30), int(self.char_height * 0.30)),
                rotation_point=(0.5, 0.9)
            ),
            'torso': BodyPart(
                name='torso',
                region=(int(self.char_width * 0.25), int(self.char_height * 0.25), int(self.char_width * 0.5), int(self.char_height * 0.5)),
                rotation_point=(0.5, 0.1)
            ),
            'right_upper_arm': BodyPart(
                name='right_upper_arm',
                region=(int(self.char_width * 0.05), int(self.char_height * 0.25), int(self.char_width * 0.25), int(self.char_height * 0.3)),
                rotation_point=(0.2, 0.2), parent='torso'
            ),
            'left_upper_arm': BodyPart(
                name='left_upper_arm',
                region=(int(self.char_width * 0.70), int(self.char_height * 0.25), int(self.char_width * 0.25), int(self.char_height * 0.3)),
                rotation_point=(0.8, 0.2), parent='torso'
            ),
            'left_forearm': BodyPart(
                name='left_forearm',
                region=(int(self.char_width * 0.00), int(self.char_height * 0.45), int(self.char_width * 0.25), int(self.char_height * 0.25)),
                rotation_point=(0.8, 0.1), parent='left_upper_arm'
            ),
            'right_forearm': BodyPart(
                name='right_forearm',
                region=(int(self.char_width * 0.75), int(self.char_height * 0.45), int(self.char_width * 0.25), int(self.char_height * 0.25)),
                rotation_point=(0.2, 0.1), parent='right_upper_arm'
            ),
            'legs': BodyPart(
                name='legs',
                region=(int(self.char_width * 0.20), int(self.char_height * 0.60), int(self.char_width * 0.60), int(self.char_height * 0.40)),
                rotation_point=(0.5, 0.1)
            ),
        }

        # Extract sub-images
        self.body_part_images = self.extract_body_parts()

        # Character reference pose landmarks (normalized)
        self.char_reference_pose = {
            'left_shoulder': (0.25, 0.28),
            'right_shoulder': (0.75, 0.28),
            'left_elbow': (0.15, 0.48),
            'right_elbow': (0.85, 0.48),
            'left_wrist': (0.12, 0.65),
            'right_wrist': (0.88, 0.65),
            'nose': (0.5, 0.15),
            'left_hip': (0.40, 0.62),
            'right_hip': (0.60, 0.62)
        }

        # Smoothing for pose estimation
        self.pose_history: List[Dict[str, Tuple[float, float]]] = []
        self.smoothing_frames = 5

        # UI state
        self.show_debug = False

    def extract_body_parts(self) -> Dict[str, np.ndarray]:
        parts = {}
        for part_name, part_info in self.body_parts.items():
            x, y, w, h = part_info.region
            # clamp to image bounds
            x = max(0, min(x, self.char_width - 1))
            y = max(0, min(y, self.char_height - 1))
            w = max(0, min(w, self.char_width - x))
            h = max(0, min(h, self.char_height - y))
            parts[part_name] = self.character_img[y:y + h, x:x + w].copy()
        return parts

    def smooth_pose(self, current_landmarks: Dict[str, Tuple[float, float]]):
        self.pose_history.append(current_landmarks)
        if len(self.pose_history) > self.smoothing_frames:
            self.pose_history.pop(0)
        if len(self.pose_history) == 1:
            return current_landmarks
        smoothed = {}
        for key in current_landmarks:
            x_sum = sum(p[key][0] for p in self.pose_history)
            y_sum = sum(p[key][1] for p in self.pose_history)
            smoothed[key] = (x_sum / len(self.pose_history), y_sum / len(self.pose_history))
        return smoothed

    @staticmethod
    def calculate_rotation_angle(p1, p2, reference_angle=0.0):
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        return angle - reference_angle

    @staticmethod
    def rotate_image_around_point(image: np.ndarray, angle: float, pivot_point: Tuple[float, float]):
        h, w = image.shape[:2]
        px, py = int(pivot_point[0] * w), int(pivot_point[1] * h)
        M = cv2.getRotationMatrix2D((px, py), math.degrees(angle), 1.0)
        cos_v, sin_v = abs(M[0, 0]), abs(M[0, 1])
        new_w = int((h * sin_v) + (w * cos_v))
        new_h = int((h * cos_v) + (w * sin_v))
        M[0, 2] += (new_w / 2) - px
        M[1, 2] += (new_h / 2) - py
        rot = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return rot, M

    def get_pose_landmarks(self, person_landmarks, frame_shape):
        h, w = frame_shape[:2]
        lm = {}
        mpL = self.mp_pose.PoseLandmark
        mapping = {
            'left_shoulder': mpL.LEFT_SHOULDER,
            'right_shoulder': mpL.RIGHT_SHOULDER,
            'left_elbow': mpL.LEFT_ELBOW,
            'right_elbow': mpL.RIGHT_ELBOW,
            'left_wrist': mpL.LEFT_WRIST,
            'right_wrist': mpL.RIGHT_WRIST,
            'nose': mpL.NOSE,
            'left_hip': mpL.LEFT_HIP,
            'right_hip': mpL.RIGHT_HIP
        }
        for name, idx in mapping.items():
            p = person_landmarks.landmark[idx]
            lm[name] = (p.x * w, p.y * h)
        return self.smooth_pose(lm)

    def calculate_body_part_transforms(self, person_landmarks: Dict[str, Tuple[float, float]]):
        transforms = {}
        shoulder_width = abs(person_landmarks['right_shoulder'][0] - person_landmarks['left_shoulder'][0])
        char_shoulder_w = abs(self.char_reference_pose['right_shoulder'][0] - self.char_reference_pose['left_shoulder'][0]) * self.char_width
        scale = max(0.01, shoulder_width * 1.75 / char_shoulder_w * 0.8)

        body_center = (
            (person_landmarks['left_shoulder'][0] + person_landmarks['right_shoulder'][0]) / 2.0,
            (person_landmarks['left_shoulder'][1] + person_landmarks['right_shoulder'][1]) / 2.0
        )

        ref_l_arm = math.atan2(
            self.char_reference_pose['left_elbow'][1] - self.char_reference_pose['left_shoulder'][1],
            self.char_reference_pose['left_elbow'][0] - self.char_reference_pose['left_shoulder'][0]
        )
        ref_r_arm = math.atan2(
            self.char_reference_pose['right_elbow'][1] - self.char_reference_pose['right_shoulder'][1],
            self.char_reference_pose['right_elbow'][0] - self.char_reference_pose['right_shoulder'][0]
        )

        l_arm = -self.calculate_rotation_angle(person_landmarks['left_shoulder'], person_landmarks['left_elbow'], ref_l_arm)
        r_arm = -self.calculate_rotation_angle(person_landmarks['right_shoulder'], person_landmarks['right_elbow'], ref_r_arm)
        l_fore = -self.calculate_rotation_angle(person_landmarks['left_elbow'], person_landmarks['left_wrist']) - l_arm
        r_fore = -self.calculate_rotation_angle(person_landmarks['right_elbow'], person_landmarks['right_wrist']) - r_arm

        transforms['left_upper_arm'] = {'rotation': l_arm, 'scale': scale, 'position': person_landmarks['left_shoulder']}
        transforms['right_upper_arm'] = {'rotation': r_arm, 'scale': scale, 'position': person_landmarks['right_shoulder']}
        transforms['left_forearm'] = {'rotation': l_fore, 'scale': scale, 'position': person_landmarks['left_elbow']}
        transforms['right_forearm'] = {'rotation': r_fore, 'scale': scale, 'position': person_landmarks['right_elbow']}
        transforms['torso'] = {'rotation': 0.0, 'scale': scale, 'position': body_center}

        nx, ny = person_landmarks['nose']
        transforms['head'] = {'rotation': 0.0, 'scale': scale, 'position': (nx, ny + 50)}
        transforms['legs'] = {
            'rotation': 0.0, 'scale': scale,
            'position': ((person_landmarks['left_hip'][0] + person_landmarks['right_hip'][0]) / 2.0,
                         (person_landmarks['left_hip'][1] + person_landmarks['right_hip'][1]) / 2.0)
        }
        return transforms

    def blend_image(self, background: np.ndarray, overlay: np.ndarray, position: Tuple[int, int]):
        x, y = position
        h, w = overlay.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # overlap bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
        if x1 >= x2 or y1 >= y2:
            return background

        ox1, oy1 = x1 - x, y1 - y
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

        if overlay.shape[2] == 4:
            alpha = overlay[oy1:oy2, ox1:ox2, 3:4] / 255.0
            ov_rgb = overlay[oy1:oy2, ox1:ox2, :3]
            bg_roi = background[y1:y2, x1:x2]
            blended = (alpha * ov_rgb + (1 - alpha) * bg_roi).astype(np.uint8)
            background[y1:y2, x1:x2] = blended
        else:
            background[y1:y2, x1:x2] = overlay[oy1:oy2, ox1:ox2]
        return background

    def render_character(self, frame: np.ndarray, person_landmarks):
        if person_landmarks is None:
            return frame

        landmarks = self.get_pose_landmarks(person_landmarks, frame.shape)
        transforms = self.calculate_body_part_transforms(landmarks)

        result = frame.copy()
        render_order = ['legs', 'torso', 'left_upper_arm', 'right_upper_arm', 'left_forearm', 'right_forearm', 'head']

        for part in render_order:
            if part not in self.body_part_images or part not in transforms:
                continue
            part_img = self.body_part_images[part]
            info = self.body_parts[part]
            t = transforms[part]

            rotated, _ = self.rotate_image_around_point(part_img, t['rotation'], info.rotation_point)
            new_size = (max(1, int(rotated.shape[1] * t['scale'])), max(1, int(rotated.shape[0] * t['scale'])))
            scaled = cv2.resize(rotated, new_size, interpolation=cv2.INTER_LINEAR)

            pos_x = int(t['position'][0] - scaled.shape[1] // 2)
            pos_y = int(t['position'][1] - scaled.shape[0] // 2)
            result = self.blend_image(result, scaled, (pos_x, pos_y))

        return result

    # ---------- NEW: recording helpers / UI ----------

    @staticmethod
    def _draw_rec_button(frame: np.ndarray, recording: bool):
        """Draws a REC/STOP button bottom-right; returns bbox (x1,y1,x2,y2)."""
        h, w = frame.shape[:2]
        btn_w, btn_h, m = 150, 50, 20
        x1, y1 = w - btn_w - m, h - btn_h - m
        x2, y2 = w - m, h - m
        color = (0, 0, 255) if recording else (50, 180, 50)
        label = "STOP" if recording else "REC"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, label, (x1 + 35, y1 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        if recording:
            cv2.circle(frame, (20, 30), 8, (0, 0, 255), -1)
        return (x1, y1, x2, y2)

    @staticmethod
    def _get_writer(frame_shape, fps=30.0, out_dir="recordings"):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"capture_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = frame_shape[:2]
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        return writer, path

    def run(self, cam_index: int = 0, window_name: str = "Advanced Character Mirror"):
        # On macOS, AVFoundation is more reliable; let OpenCV choose best backend if not available
        cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" else cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open camera. Check macOS camera permissions and try another index.")

        # derive FPS (fallback to 30)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 30.0

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # mouse click state for the button
        clicked = {"flag": False}
        btn_rect = (0, 0, 0, 0)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                x1, y1, x2, y2 = btn_rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    clicked["flag"] = True

        cv2.setMouseCallback(window_name, on_mouse)

        recording, writer, out_path = False, None, None

        print("Controls: q=quit, d=toggle landmarks, s=save frame, v=record toggle, click button=record toggle")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Frame grab failed.")
                    break

                frame = cv2.flip(frame, 1)

                # Pose
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb)
                if results.pose_landmarks:
                    frame = self.render_character(frame, results.pose_landmarks)
                    if self.show_debug:
                        self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # Draw button on top
                btn_rect = self._draw_rec_button(frame, recording)

                # Write if recording
                if recording and writer is not None:
                    writer.write(frame)

                # Show
                cv2.imshow(window_name, frame)

                # Handle click toggle
                if clicked["flag"]:
                    clicked["flag"] = False
                    if not recording:
                        writer, out_path = self._get_writer(frame.shape, fps)
                        recording = True
                        print(f"⏺ Recording started → {out_path}")
                    else:
                        recording = False
                        if writer is not None:
                            writer.release()
                            writer = None
                            print(f"✅ Recording saved → {out_path}")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"Debug: {'ON' if self.show_debug else 'OFF'}")
                elif key == ord('s'):
                    snap = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.jpg")
                    cv2.imwrite(snap, frame)
                    print(f"Saved {snap}")
                elif key == ord('v'):
                    if not recording:
                        writer, out_path = self._get_writer(frame.shape, fps)
                        recording = True
                        print(f"⏺ Recording started → {out_path}")
                    else:
                        recording = False
                        if writer is not None:
                            writer.release()
                            writer = None
                            print(f"✅ Recording saved → {out_path}")

        finally:
            cap.release()
            if writer is not None:
                writer.release()
                print(f"✅ Recording saved → {out_path}")
            cv2.destroyAllWindows()


def main():
    # Replace with your PNG with alpha (RGBA). Keep it near the script for easy testing.
    character_image_path = os.path.join(os.path.dirname(__file__), "Human_body_outline.png")
    if not os.path.isfile(character_image_path):
        print(f"⚠ Character image not found at: {character_image_path}")
        print("   Place an RGBA PNG there or update the path in main().")
    try:
        app = AdvancedCharacterPoseMirror(character_image_path)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nEnsure:")
        print("  1) An RGBA character image (PNG) exists at the path above.")
        print("  2) Packages installed: pip install opencv-python mediapipe numpy")
        print("  3) Camera permissions are granted (macOS: System Settings → Privacy & Security → Camera).")


if __name__ == "__main__":
    main()
