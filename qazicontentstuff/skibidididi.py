"""
Bulletproof 3D Object Tracker with Enhanced .obj File Support and 90° Rotation
- Works on Python 3.12+ and 3.13
- Only requires: opencv-python, mediapipe, numpy
- No OpenGL, ModernGL, or any 3D rendering library dependencies
- Enhanced .obj file loading with proper face rendering
- Added 90-degree rotation support for loaded models
- Handles all compatibility issues gracefully
"""

import cv2
import numpy as np
import os
import sys
import glob

# Handle MediaPipe import gracefully
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe loaded successfully")
except ImportError as e:
    print(f"⚠ MediaPipe not available: {e}")
    print("Install with: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

class FallbackPoseDetector:
    """Fallback pose detection using OpenCV when MediaPipe fails"""
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.frame_count = 0
        
    def detect_pose(self, frame):
        """Simple pose detection using background subtraction and contours"""
        self.frame_count += 1
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find largest contour (person)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Minimum area threshold (adjust based on distance from camera)
        if area < 8000:
            return None
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Estimate pose landmarks based on human proportions
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Human body proportions (approximate)
        head_top = y
        shoulder_y = y + int(h * 0.15)
        chest_y = y + int(h * 0.35)
        hip_y = y + int(h * 0.65)
        
        shoulder_width = int(w * 0.35)
        hip_width = int(w * 0.25)
        
        landmarks = {
            'LEFT_SHOULDER': (center_x - shoulder_width, shoulder_y),
            'RIGHT_SHOULDER': (center_x + shoulder_width, shoulder_y),
            'LEFT_HIP': (center_x - hip_width, hip_y),
            'RIGHT_HIP': (center_x + hip_width, hip_y),
            'CHEST_CENTER': (center_x, chest_y),
        }
        
        return landmarks

class MediaPipePoseDetector:
    """MediaPipe-based pose detection"""
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available")
            
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_pose(self, frame):
        """Detect pose using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
            
        h, w = frame.shape[:2]
        landmarks = {}
        
        def get_landmark(landmark_type):
            lm = results.pose_landmarks.landmark[landmark_type]
            if lm.visibility > 0.5:
                return (int(lm.x * w), int(lm.y * h))
            return None
        
        ls = get_landmark(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        rs = get_landmark(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        lh = get_landmark(self.mp_pose.PoseLandmark.LEFT_HIP)
        rh = get_landmark(self.mp_pose.PoseLandmark.RIGHT_HIP)
        
        if all([ls, rs, lh, rh]):
            shoulder_center = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
            hip_center = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
            chest_center = ((shoulder_center[0] + hip_center[0]) // 2,
                           (shoulder_center[1] + hip_center[1]) // 2)
            
            landmarks = {
                'LEFT_SHOULDER': ls,
                'RIGHT_SHOULDER': rs,
                'LEFT_HIP': lh,
                'RIGHT_HIP': rh,
                'CHEST_CENTER': chest_center,
            }
        
        return landmarks, results.pose_landmarks

class OBJLoader:
    """Enhanced .obj file loader with proper face parsing"""
    @staticmethod
    def load_obj(filepath):
        """Load .obj file and return vertices, faces, and normals"""
        vertices = []
        faces = []
        normals = []
        texture_coords = []
        
        print(f"🔄 Loading .obj file: {filepath}")
        
        try:
            with open(filepath, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    try:
                        if parts[0] == 'v':  # Vertex
                            if len(parts) >= 4:
                                coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                                vertices.append(coords)
                        
                        elif parts[0] == 'vn':  # Vertex normal
                            if len(parts) >= 4:
                                normal = [float(parts[1]), float(parts[2]), float(parts[3])]
                                normals.append(normal)
                        
                        elif parts[0] == 'vt':  # Texture coordinate
                            if len(parts) >= 3:
                                tex_coord = [float(parts[1]), float(parts[2])]
                                texture_coords.append(tex_coord)
                        
                        elif parts[0] == 'f':  # Face
                            if len(parts) >= 4:  # At least 3 vertices for a face
                                face_vertices = []
                                for vertex_data in parts[1:]:
                                    # Handle different face formats: v, v/vt, v/vt/vn, v//vn
                                    vertex_parts = vertex_data.split('/')
                                    vertex_index = int(vertex_parts[0]) - 1  # OBJ uses 1-based indexing
                                    
                                    # Ensure vertex index is valid
                                    if 0 <= vertex_index < len(vertices):
                                        face_vertices.append(vertex_index)
                                
                                # Convert to triangles if needed (for quads and polygons)
                                if len(face_vertices) >= 3:
                                    # For triangles
                                    if len(face_vertices) == 3:
                                        faces.append(face_vertices)
                                    # For quads, create two triangles
                                    elif len(face_vertices) == 4:
                                        faces.append([face_vertices[0], face_vertices[1], face_vertices[2]])
                                        faces.append([face_vertices[0], face_vertices[2], face_vertices[3]])
                                    # For n-gons, fan triangulation
                                    else:
                                        for i in range(1, len(face_vertices) - 1):
                                            faces.append([face_vertices[0], face_vertices[i], face_vertices[i + 1]])
                    
                    except (ValueError, IndexError) as e:
                        print(f"⚠ Warning: Error parsing line {line_num}: '{line}' - {e}")
                        continue
        
        except Exception as e:
            print(f"❌ Error loading .obj file: {e}")
            return None, None
        
        if not vertices or not faces:
            print(f"❌ Invalid .obj file: {len(vertices)} vertices, {len(faces)} faces")
            return None, None
        
        print(f"✓ Successfully loaded: {len(vertices)} vertices, {len(faces)} faces")
        if normals:
            print(f"✓ Loaded {len(normals)} normals")
        if texture_coords:
            print(f"✓ Loaded {len(texture_coords)} texture coordinates")
        
        return np.array(vertices, dtype=np.float32), faces

class ChestPlate3DModel:
    """3D chest plate model with enhanced .obj loading and rotation support"""
    def __init__(self, obj_path=None, rotation_angle=90):
        self.vertices = None
        self.faces = []
        self.colors = []
        self.obj_loaded = False
        self.model_name = "Default"
        self.rotation_angle = rotation_angle
        
        if obj_path:
            self.load_obj_file(obj_path)
        
        if not self.obj_loaded:
            self.create_default_model()
        
        self.generate_face_colors()
        
    def find_obj_files(self):
        """Find all .obj files in current directory"""
        obj_files = glob.glob("*.obj")
        return obj_files
    
    def apply_rotation(self, vertices, angle_degrees, axis='y'):
        """Apply rotation to vertices around specified axis"""
        angle_rad = np.radians(angle_degrees)
        
        if axis == 'x':
            # Rotation around X-axis
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == 'y':
            # Rotation around Y-axis
            rotation_matrix = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        elif axis == 'z':
            # Rotation around Z-axis
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        else:
            print(f"⚠ Unknown rotation axis: {axis}, using Y-axis")
            return self.apply_rotation(vertices, angle_degrees, 'y')
        
        # Apply rotation to all vertices
        rotated_vertices = np.dot(vertices, rotation_matrix.T)
        print(f"🔄 Applied {angle_degrees}° rotation around {axis.upper()}-axis")
        return rotated_vertices
    
    def load_obj_file(self, obj_path):
        """Load .obj file with enhanced error handling and rotation"""
        # First try exact path
        if os.path.exists(obj_path):
            vertices, faces = OBJLoader.load_obj(obj_path)
            if vertices is not None and faces is not None:
                self.vertices = vertices
                self.faces = faces
                self.obj_loaded = True
                self.model_name = os.path.basename(obj_path)
                print(f"✓ Loaded model: {self.model_name}")
                
                # Apply rotation if specified
                if self.rotation_angle != 0:
                    self.vertices = self.apply_rotation(self.vertices, self.rotation_angle)
                
                self.normalize_model()
                return
        
        # Try to find .obj files in current directory
        obj_files = self.find_obj_files()
        if obj_files:
            print(f"🔍 Found .obj files: {obj_files}")
            for obj_file in obj_files:
                print(f"🔄 Trying to load: {obj_file}")
                vertices, faces = OBJLoader.load_obj(obj_file)
                if vertices is not None and faces is not None:
                    self.vertices = vertices
                    self.faces = faces
                    self.obj_loaded = True
                    self.model_name = obj_file
                    print(f"✓ Successfully loaded: {obj_file}")
                    
                    # Apply rotation if specified
                    if self.rotation_angle != 0:
                        self.vertices = self.apply_rotation(self.vertices, self.rotation_angle)
                    
                    self.normalize_model()
                    return
        
        print(f"⚠ Could not load .obj file: {obj_path}")
        print("Available .obj files:", obj_files if obj_files else "None found")
    
    def normalize_model(self):
        """Normalize model to fit chest area"""
        if self.vertices is None or len(self.vertices) == 0:
            return
        
        # Calculate bounding box
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        
        print(f"📏 Model bounds: min={min_coords}, max={max_coords}")
        
        # Center the model
        center = (min_coords + max_coords) / 2
        self.vertices -= center
        
        # Scale to appropriate size for chest area
        size = max_coords - min_coords
        max_dimension = np.max(size)
        
        if max_dimension > 0:
            # Scale to fit roughly 20cm chest plate
            target_size = 0.2
            scale_factor = target_size / max_dimension
            self.vertices *= scale_factor
            print(f"📏 Scaled model by factor: {scale_factor:.3f}")
        
        print(f"✓ Model normalized. Final size: {np.max(self.vertices) - np.min(self.vertices):.3f}")
    
    def create_default_model(self):
        """Create enhanced default chest plate model"""
        print("🔧 Creating default chest plate model")
        
        vertices = []
        
        # Main chest plate (trapezoid with rounded edges)
        width_top = 0.14
        width_bottom = 0.11
        height = 0.18
        depth = 0.03
        
        # Front face vertices
        vertices.extend([
            [-width_top, height/2, depth],      # 0: Top left front
            [width_top, height/2, depth],       # 1: Top right front
            [width_bottom, -height/2, depth],   # 2: Bottom right front
            [-width_bottom, -height/2, depth],  # 3: Bottom left front
        ])
        
        # Back face vertices (slightly smaller)
        back_scale = 0.9
        vertices.extend([
            [-width_top * back_scale, height/2 * back_scale, -depth],    # 4: Top left back
            [width_top * back_scale, height/2 * back_scale, -depth],     # 5: Top right back
            [width_bottom * back_scale, -height/2 * back_scale, -depth], # 6: Bottom right back
            [-width_bottom * back_scale, -height/2 * back_scale, -depth],# 7: Bottom left back
        ])
        
        # Add detail elements (center emblem)
        emblem_size = 0.04
        vertices.extend([
            [-emblem_size, emblem_size, depth + 0.005],    # 8: Emblem top left
            [emblem_size, emblem_size, depth + 0.005],     # 9: Emblem top right
            [emblem_size, -emblem_size, depth + 0.005],    # 10: Emblem bottom right
            [-emblem_size, -emblem_size, depth + 0.005],   # 11: Emblem bottom left
        ])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        
        # Define faces for the chest plate
        self.faces = [
            # Main chest plate faces
            [0, 1, 2], [0, 2, 3],  # Front face
            [4, 6, 5], [4, 7, 6],  # Back face
            [0, 4, 5], [0, 5, 1],  # Top edge
            [3, 2, 6], [3, 6, 7],  # Bottom edge
            [0, 3, 7], [0, 7, 4],  # Left edge
            [1, 5, 6], [1, 6, 2],  # Right edge
            
            # Center emblem
            [8, 9, 10], [8, 10, 11],  # Emblem face
        ]
        
        self.model_name = "Default Chest Plate"
        print(f"✓ Created default model with {len(self.vertices)} vertices and {len(self.faces)} faces")
    
    def generate_face_colors(self):
        """Generate colors for each face"""
        self.colors = []
        
        if self.obj_loaded:
            # Use varied metallic colors for loaded models
            base_colors = [
                np.array([180, 140, 60]),   # Gold
                np.array([160, 160, 160]),  # Silver
                np.array([120, 80, 40]),    # Bronze
                np.array([200, 100, 100]),  # Rose gold
            ]
        else:
            # Use specific colors for default model
            base_colors = [np.array([200, 150, 50])]  # Golden base
        
        for i in range(len(self.faces)):
            if self.obj_loaded:
                # Cycle through metallic colors with variations
                base_color = base_colors[i % len(base_colors)]
                variation = (i * 10) % 40 - 20
                color = np.clip(base_color + variation, 0, 255)
            else:
                # Default model coloring
                if i < 12:  # Main chest plate
                    base_color = np.array([200, 150, 50])  # Golden
                    variation = (i * 15) % 50 - 25
                    color = np.clip(base_color + variation, 0, 255)
                else:  # Emblem
                    color = np.array([150, 100, 200])  # Purple emblem
            
            self.colors.append(tuple(map(int, color)))

class Advanced3DRenderer:
    """Advanced 3D rendering using only OpenCV"""
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
    def render_object(self, frame, model, rvec, tvec):
        """Render 3D object with advanced shading and error checking"""
        # Validate model data
        if model.vertices is None or len(model.vertices) == 0 or len(model.faces) == 0:
            print("⚠ No valid model data to render")
            return
        
        # Project all vertices to 2D
        try:
            projected_points, _ = cv2.projectPoints(
                model.vertices, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2).astype(int)
        except Exception as e:
            print(f"⚠ Projection error: {e}")
            return
        
        # Calculate face depths and normals for sorting and lighting
        face_data = []
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        for i, face in enumerate(model.faces):
            if len(face) < 3:
                continue
                
            # Validate face indices
            if not all(0 <= idx < len(model.vertices) for idx in face):
                print(f"⚠ Invalid face indices {face}, skipping")
                continue
            
            try:
                # Calculate face center and normal
                face_vertices = [model.vertices[j] for j in face[:3]]
                center_3d = np.mean(face_vertices, axis=0)
                
                # Calculate normal
                v1, v2, v3 = face_vertices
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                normal_length = np.linalg.norm(normal)
                if normal_length > 0:
                    normal = normal / normal_length
                else:
                    continue  # Skip degenerate face
                
                # Transform to camera space
                center_cam = rotation_matrix @ center_3d + tvec.flatten()
                normal_cam = rotation_matrix @ normal
                
                # Backface culling - only render faces facing camera
                view_direction = np.array([0, 0, -1])  # Camera looks down negative Z
                if np.dot(normal_cam, view_direction) < 0:
                    continue  # Skip back-facing faces
                
                # Calculate lighting
                light_dir = np.array([0.3, -0.5, -1])  # Light from front-top
                light_dir = light_dir / np.linalg.norm(light_dir)
                intensity = max(0.3, min(1.0, np.dot(normal_cam, light_dir)))
                
                face_data.append({
                    'face': face,
                    'depth': center_cam[2],
                    'intensity': intensity,
                    'color': model.colors[i] if i < len(model.colors) else (150, 150, 150)
                })
                
            except Exception as e:
                print(f"⚠ Error processing face {i}: {e}")
                continue
        
        if not face_data:
            return
        
        # Sort faces by depth (back to front for proper alpha blending)
        face_data.sort(key=lambda x: x['depth'], reverse=True)
        
        # Render faces
        h, w = frame.shape[:2]
        for data in face_data:
            face = data['face']
            intensity = data['intensity']
            base_color = np.array(data['color'])
            
            # Apply lighting
            lit_color = (base_color * intensity).astype(int)
            lit_color = np.clip(lit_color, 0, 255)
            color = tuple(map(int, lit_color))
            
            # Get 2D face points with bounds checking
            face_points_2d = []
            for vertex_idx in face:
                if 0 <= vertex_idx < len(projected_points):
                    face_points_2d.append(projected_points[vertex_idx])
                else:
                    break
            
            if len(face_points_2d) >= 3:
                face_points_2d = np.array(face_points_2d, dtype=np.int32)
                
                # Check if points are within reasonable bounds (allow some off-screen)
                valid_points = []
                for point in face_points_2d:
                    x, y = point
                    # Allow points slightly outside frame for partial visibility
                    x = max(-100, min(w + 100, x))
                    y = max(-100, min(h + 100, y))
                    valid_points.append([x, y])
                
                if len(valid_points) >= 3:
                    valid_points = np.array(valid_points, dtype=np.int32)
                    
                    try:
                        # Create overlay for transparency
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [valid_points], color)
                        
                        # Blend with original frame (more opaque for loaded models)
                        alpha = 0.85 if model.obj_loaded else 0.75
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        
                        # Draw edges for definition
                        edge_color = tuple(min(255, max(0, c + 30)) for c in color)
                        cv2.polylines(frame, [valid_points], True, edge_color, 1)
                        
                    except Exception as e:
                        print(f"⚠ Rendering error: {e}")
    
    def draw_debug_info(self, frame, rvec, tvec):
        """Draw coordinate axes and debug information"""
        # Coordinate axes
        axes_length = 0.08
        axes_points = np.array([
            [0, 0, 0],                    # Origin
            [axes_length, 0, 0],          # X axis
            [0, axes_length, 0],          # Y axis
            [0, 0, axes_length]           # Z axis
        ], dtype=np.float32)
        
        try:
            projected_axes, _ = cv2.projectPoints(
                axes_points, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
            projected_axes = projected_axes.reshape(-1, 2).astype(int)
            
            origin = tuple(projected_axes[0])
            x_axis = tuple(projected_axes[1])
            y_axis = tuple(projected_axes[2])
            z_axis = tuple(projected_axes[3])
            
            # Draw axes with labels
            cv2.arrowedLine(frame, origin, x_axis, (0, 0, 255), 3, tipLength=0.3)  # Red X
            cv2.arrowedLine(frame, origin, y_axis, (0, 255, 0), 3, tipLength=0.3)  # Green Y
            cv2.arrowedLine(frame, origin, z_axis, (255, 0, 0), 3, tipLength=0.3)  # Blue Z
            
            # Axis labels
            cv2.putText(frame, 'X', x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, 'Y', y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, 'Z', z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except Exception as e:
            print(f"⚠ Debug rendering error: {e}")

class BulletproofObjectTracker:
    """Main tracker class with fallback mechanisms"""
    def __init__(self, obj_path=None, rotation_angle=90):
        print("🚀 Initializing Bulletproof 3D Object Tracker")
        
        # Try MediaPipe first, fallback to OpenCV
        try:
            self.pose_detector = MediaPipePoseDetector()
            self.use_mediapipe = True
            print("✓ Using MediaPipe pose detection")
        except Exception as e:
            print(f"⚠ MediaPipe failed: {e}")
            self.pose_detector = FallbackPoseDetector()
            self.use_mediapipe = False
            print("✓ Using fallback OpenCV pose detection")
        
        # Load 3D model with rotation
        self.model = ChestPlate3DModel(obj_path, rotation_angle)
        
        # Camera parameters (adjust based on your camera)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))
        
        # Initialize renderer
        self.renderer = Advanced3DRenderer(self.camera_matrix, self.dist_coeffs)
        
        # State
        self.person_detected = False
        self.pose_landmarks = None
        self.frame_count = 0
        self.show_debug = True
        self.rotation_angle = rotation_angle
        
        print(f"✅ Tracker initialized with model: {self.model.model_name}")
        if rotation_angle != 0:
            print(f"🔄 Model rotated by {rotation_angle}°")
    
    def estimate_3d_pose(self, landmarks):
        """Estimate 3D pose using PnP solver"""
        required_points = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP', 'CHEST_CENTER']
        
        if not all(point in landmarks for point in required_points):
            return None, None
        
        # 3D model reference points (human body proportions)
        model_points = np.array([
            [0.0, 0.0, 0.0],       # Chest center
            [-0.15, 0.1, 0.0],     # Left shoulder
            [0.15, 0.1, 0.0],      # Right shoulder
            [-0.1, -0.15, 0.0],    # Left hip
            [0.1, -0.15, 0.0],     # Right hip
        ], dtype=np.float32)
        
        # Corresponding 2D image points
        image_points = np.array([
            landmarks['CHEST_CENTER'],
            landmarks['LEFT_SHOULDER'],
            landmarks['RIGHT_SHOULDER'],
            landmarks['LEFT_HIP'],
            landmarks['RIGHT_HIP']
        ], dtype=np.float32)
        
        # Solve PnP with error handling
        try:
            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                return rvec, tvec
            
        except Exception as e:
            print(f"⚠ PnP solver error: {e}")
        
        return None, None
    
    def draw_landmarks(self, frame, landmarks):
        """Draw detected landmarks"""
        colors = {
            'LEFT_SHOULDER': (255, 0, 0),
            'RIGHT_SHOULDER': (255, 0, 0),
            'LEFT_HIP': (0, 255, 0),
            'RIGHT_HIP': (0, 255, 0),
            'CHEST_CENTER': (0, 255, 255)
        }
        
        for name, pos in landmarks.items():
            color = colors.get(name, (255, 255, 255))
            cv2.circle(frame, tuple(map(int, pos)), 6, color, -1)
            cv2.circle(frame, tuple(map(int, pos)), 8, (255, 255, 255), 2)
            
            # Label
            cv2.putText(frame, name.replace('_', ' '), 
                       (int(pos[0]) + 10, int(pos[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def process_frame(self, frame):
        """Main frame processing"""
        self.frame_count += 1
        
        # Detect pose
        if self.use_mediapipe:
            result = self.pose_detector.detect_pose(frame)
            if result and len(result) == 2:
                landmarks, self.pose_landmarks = result
            else:
                landmarks = None
        else:
            landmarks = self.pose_detector.detect_pose(frame)
        
        if landmarks:
            self.person_detected = True
            
            # Draw MediaPipe skeleton if available
            if self.use_mediapipe and self.pose_landmarks:
                mp_drawing = mp.solutions.drawing_utils
                mp_pose = mp.solutions.pose
                mp_drawing.draw_landmarks(
                    frame, self.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
            
            # Draw landmark points
            self.draw_landmarks(frame, landmarks)
            
            # Estimate 3D pose and render
            rvec, tvec = self.estimate_3d_pose(landmarks)
            
            if rvec is not None and tvec is not None:
                # Render 3D object
                self.renderer.render_object(frame, self.model, rvec, tvec)
                
                # Draw debug info
                if self.show_debug:
                    self.renderer.draw_debug_info(frame, rvec, tvec)
        else:
            self.person_detected = False
        
        return frame


def main():
    """Main function with comprehensive error handling"""
    print("=" * 60)
    print("🛡️  BULLETPROOF 3D OBJECT TRACKER WITH 90° ROTATION")
    print("=" * 60)
    print("Enhanced .obj file support for armor/chest plate models")
    print("Compatible with Python 3.12+ and 3.13")
    print("Automatic fallbacks for all compatibility issues")
    print("🔄 NEW: Built-in 90° rotation for loaded .obj models")
    print()
    
    # Check dependencies
    print("📋 Checking dependencies:")
    print(f"✓ OpenCV version: {cv2.__version__}")
    print(f"✓ NumPy version: {np.__version__}")
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    if MEDIAPIPE_AVAILABLE:
        print(f"✓ MediaPipe version: {mp.__version__}")
    else:
        print("⚠ MediaPipe not available - using OpenCV fallback")
    
    print()
    
    # Check for .obj files in current directory
    print("🔍 Searching for .obj files in current directory...")
    obj_files = glob.glob("*.obj")
    if obj_files:
        print(f"✓ Found .obj files: {obj_files}")
        obj_path = obj_files[0]  # Use first found .obj file
        print(f"📁 Will attempt to load: {obj_path}")
    else:
        print("⚠ No .obj files found in current directory")
        print("💡 Place your armor .obj file in the same directory as this script")
        obj_path = None
    
    print()
    
    # Test camera
    print("📹 Testing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        print("Please check your camera connection and permissions")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    ret, test_frame = cap.read()
    if ret:
        print(f"✓ Camera working! Frame size: {test_frame.shape}")
    else:
        print("❌ Error: Could not read from camera")
        cap.release()
        return
    
    print()
    
    # Initialize tracker with 90-degree rotation
    rotation_angle = 90  # Change this value to adjust rotation
    tracker = BulletproofObjectTracker(obj_path, rotation_angle)
    
    print()
    print("🎮 Controls:")
    print("  Q - Quit")
    print("  D - Toggle debug info (coordinate axes)")
    print("  R - Reset background (for fallback mode)")
    print("  L - List available .obj files")
    print("  H - Show help")
    print("  1 - No rotation (0°)")
    print("  2 - 90° rotation")
    print("  3 - 180° rotation")
    print("  4 - 270° rotation")
    print("  X/Y/Z - Change rotation axis (X, Y, or Z)")
    print()
    print("🚀 Starting tracker... Stand in front of camera!")
    print(f"📦 Current model: {tracker.model.model_name}")
    print(f"🎨 Model type: {'Loaded .obj file' if tracker.model.obj_loaded else 'Default generated model'}")
    print(f"🔄 Current rotation: {rotation_angle}° around Y-axis")
    print()
    
    # Rotation settings
    current_rotation = rotation_angle
    current_axis = 'y'
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Mirror effect for better user experience
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame = tracker.process_frame(frame)
            
            # Status overlay
            status_text = "🎯 3D ARMOR TRACKING ACTIVE!" if tracker.person_detected else "👤 PERSON NOT DETECTED"
            status_color = (0, 255, 0) if tracker.person_detected else (0, 100, 255)
            
            # Enhanced status background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (580, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame, (10, 10), (580, 140), status_color, 2)
            
            cv2.putText(frame, status_text, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Mode and model info
            mode_text = "MediaPipe Mode" if tracker.use_mediapipe else "Fallback Mode"
            cv2.putText(frame, f"Detection: {mode_text}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            model_text = f"Model: {tracker.model.model_name[:25]}"
            cv2.putText(frame, model_text, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Frame: {tracker.frame_count}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Rotation info
            rotation_text = f"Rotation: {current_rotation}° ({current_axis.upper()}-axis)"
            cv2.putText(frame, rotation_text, (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Model stats in bottom right
            if tracker.model.obj_loaded:
                stats_text = f"Vertices: {len(tracker.model.vertices)} | Faces: {len(tracker.model.faces)}"
                cv2.putText(frame, stats_text, 
                           (frame.shape[1] - 250, frame.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Instructions for fallback mode
            if not tracker.use_mediapipe and tracker.frame_count < 50:
                cv2.putText(frame, "Move around to initialize background detection", 
                           (10, frame.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Instruction for .obj files and rotation
            if not tracker.model.obj_loaded and tracker.frame_count < 100:
                cv2.putText(frame, "Place your armor.obj file in this directory for custom models", 
                           (10, frame.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)
            
            # Rotation instructions
            cv2.putText(frame, "Press 1-4 for rotation, X/Y/Z for axis", 
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
            
            cv2.imshow("Enhanced 3D Object Tracker - 90° Rotation Support", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key == ord('d'):
                tracker.show_debug = not tracker.show_debug
                print(f"🔧 Debug info: {'ON' if tracker.show_debug else 'OFF'}")
            elif key == ord('r'):
                if not tracker.use_mediapipe:
                    tracker.pose_detector = FallbackPoseDetector()
                    print("🔄 Background reset")
            elif key == ord('l'):
                print("\n📁 Available .obj files:")
                obj_files = glob.glob("*.obj")
                if obj_files:
                    for i, obj_file in enumerate(obj_files):
                        status = " (LOADED)" if obj_file == tracker.model.model_name else ""
                        print(f"  {i+1}. {obj_file}{status}")
                else:
                    print("  No .obj files found in current directory")
                print()
            elif key == ord('h'):
                print("\n🎮 Controls:")
                print("  Q - Quit")
                print("  D - Toggle debug info (coordinate axes)")
                print("  R - Reset background (fallback mode)")
                print("  L - List available .obj files") 
                print("  H - Show this help")
                print("  1 - No rotation (0°)")
                print("  2 - 90° rotation")
                print("  3 - 180° rotation") 
                print("  4 - 270° rotation")
                print("  X/Y/Z - Change rotation axis")
                print("\n💡 Tips:")
                print("  - Place .obj files in the same directory as this script")
                print("  - Supported formats: Triangulated faces, quads, n-gons")
                print("  - Model will be automatically normalized to chest size")
                print("  - Works best with armor/chest plate models")
                print("  - Try different rotation angles if model appears sideways")
                print()
            # Rotation controls
            elif key == ord('1'):
                current_rotation = 0
                tracker = BulletproofObjectTracker(obj_path, current_rotation)
                print(f"🔄 Rotation set to {current_rotation}° around {current_axis.upper()}-axis")
            elif key == ord('2'):
                current_rotation = 90
                tracker = BulletproofObjectTracker(obj_path, current_rotation)
                print(f"🔄 Rotation set to {current_rotation}° around {current_axis.upper()}-axis")
            elif key == ord('3'):
                current_rotation = 180
                tracker = BulletproofObjectTracker(obj_path, current_rotation)
                print(f"🔄 Rotation set to {current_rotation}° around {current_axis.upper()}-axis")
            elif key == ord('4'):
                current_rotation = 270
                tracker = BulletproofObjectTracker(obj_path, current_rotation)
                print(f"🔄 Rotation set to {current_rotation}° around {current_axis.upper()}-axis")
            elif key == ord('x'):
                current_axis = 'x'
                # Apply rotation around new axis
                if tracker.model.obj_loaded and current_rotation != 0:
                    tracker = BulletproofObjectTracker(obj_path, current_rotation)
                    # Update the rotation axis in the model
                    if tracker.model.vertices is not None:
                        original_vertices = tracker.model.vertices.copy()
                        tracker.model.vertices = tracker.model.apply_rotation(
                            original_vertices, current_rotation, current_axis
                        )
                print(f"🔄 Rotation axis changed to {current_axis.upper()}")
            elif key == ord('y'):
                current_axis = 'y'
                # Apply rotation around new axis
                if tracker.model.obj_loaded and current_rotation != 0:
                    tracker = BulletproofObjectTracker(obj_path, current_rotation)
                print(f"🔄 Rotation axis changed to {current_axis.upper()}")
            elif key == ord('z'):
                current_axis = 'z'
                # Apply rotation around new axis
                if tracker.model.obj_loaded and current_rotation != 0:
                    tracker = BulletproofObjectTracker(obj_path, current_rotation)
                    # Update the rotation axis in the model
                    if tracker.model.vertices is not None:
                        original_vertices = tracker.model.vertices.copy()
                        tracker.model.vertices = tracker.model.apply_rotation(
                            original_vertices, current_rotation, current_axis
                        )
                print(f"🔄 Rotation axis changed to {current_axis.upper()}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup complete")
        print("👋 Thanks for using Enhanced 3D Object Tracker!")
        
        # Final model info
        if tracker and tracker.model:
            print(f"\n📊 Session Summary:")
            print(f"Model used: {tracker.model.model_name}")
            print(f"Model type: {'Custom .obj' if tracker.model.obj_loaded else 'Default'}")
            print(f"Final rotation: {current_rotation}° around {current_axis.upper()}-axis")
            if tracker.model.obj_loaded:
                print(f"Vertices: {len(tracker.model.vertices)}")
                print(f"Faces: {len(tracker.model.faces)}")
            print(f"Frames processed: {tracker.frame_count}")


if __name__ == "__main__":
    main()