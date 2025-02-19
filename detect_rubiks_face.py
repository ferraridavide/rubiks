import cv2
import numpy as np
import argparse
import time
from enum import Enum
from collections import defaultdict
from detect_aruco_cam import camera_matrix, dist_coeffs, draw_marker_info

# Define the Rubik’s cube colors as an Enum.
class Color(Enum):
    RED = 'red'
    BLUE = 'blue'
    ORANGE = 'orange'
    YELLOW = 'yellow'
    WHITE = 'white'
    GREEN = 'green'
    
    def __str__(self):
        return self.value

# Mapping from marker id to its associated color (as a Color enum)
MARKER_TO_COLOR = {
    0: Color.RED,
    1: Color.BLUE,
    2: Color.ORANGE,
    3: Color.YELLOW,
    4: Color.WHITE,
    5: Color.GREEN
}

# Mapping from detected face (via its center color) to the intended cube face.
# Note: We will use the following cube face letters: U, R, F, D, L, B
color_to_face = {
    Color.WHITE:  'UP',    # Up face (U)
    Color.RED:    'RIGHT', # Right face (R)
    Color.GREEN:  'FACE',  # Front face (F)
    Color.YELLOW: 'DOWN',  # Down face (D)
    Color.ORANGE: 'LEFT',  # Left face (L)
    Color.BLUE:   'BACK'   # Back face (B)
}

# This dictionary converts our Color enum directly to the cube string letter.
color_to_letter = {
    Color.WHITE: 'U',
    Color.RED: 'R',
    Color.GREEN: 'F',
    Color.YELLOW: 'D',
    Color.ORANGE: 'L',
    Color.BLUE: 'B'
}

# Global variables for calibration
calibrated_colors = {}  # Will store Lab values keyed by Color enum.
current_color = None
calibration_complete = False
marker_size = 0.05  # 5cm marker
cell_size = marker_size * 1.5

# Helper function: Convert a BGR color (or average) to Lab.
def bgr_to_lab(bgr):
    """
    Converts a BGR color (as a tuple or list of 3 ints) to Lab.
    """
    # cv2.cvtColor requires an image of shape (1,1,3) of type uint8.
    bgr_array = np.uint8([[list(bgr)]])
    lab = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2LAB)
    return lab[0][0].astype(np.float32)

class RubiksCubeFace:
    def __init__(self, marker_id):
        self.marker_id = marker_id
        self.samples = []         # Each sample is a list of 9 Lab values (one per cell)
        self.final_colors = None  # Will be a list of 9 Color enum values after averaging
        self.sampling_times = []
        self.start_time = None
        # Use the center color from the marker mapping.
        self.center_color = MARKER_TO_COLOR[marker_id]
        
    def add_sample(self, colors, lab_values):
        """Store both the detected color names and their Lab values."""
        self.samples.append(lab_values)  # each sample is a list of 9 Lab values
        self.sampling_times.append(time.time())
        
    def is_sampling_complete(self):
        return len(self.samples) >= 5
        
    def compute_final_colors(self):
        if not self.is_sampling_complete():
            return False
            
        # Convert samples to a numpy array for easier processing.
        all_samples = np.array(self.samples)  # shape: (num_samples, 9, 3)
        # Compute mean Lab values for each cell (resulting in an array of shape (9, 3)).
        mean_lab_values = np.mean(all_samples, axis=0)
        
        # Determine the final color for each cell.
        self.final_colors = []
        for i, lab_value in enumerate(mean_lab_values):
            # For the center position (index 4) we already know the color.
            if i == 4:
                self.final_colors.append(self.center_color)
            else:
                # Find the calibrated color whose Lab value is closest to lab_value.
                min_dist = float('inf')
                closest_color = None
                for color_name, calibrated_lab in calibrated_colors.items():
                    dist = np.linalg.norm(lab_value - calibrated_lab)
                    if dist < min_dist:
                        min_dist = dist
                        closest_color = color_name
                self.final_colors.append(closest_color)
        return True

def mouse_callback(event, x, y, flags, param):
    global calibrated_colors, current_color, calibration_complete
    if event == cv2.EVENT_LBUTTONDOWN and current_color is not None:
        frame = param
        # Ensure we are not near the border.
        h, w = frame.shape[:2]
        if x - 2 < 0 or x + 3 > w or y - 2 < 0 or y + 3 > h:
            print("Clicked too close to border; try again.")
            return
        # Sample color from a 5x5 region around the clicked point.
        roi = frame[y-2:y+3, x-2:x+3]
        avg_bgr = np.mean(roi, axis=(0, 1))
        # Convert the average BGR value to Lab.
        avg_lab = bgr_to_lab(avg_bgr)
        calibrated_colors[current_color] = avg_lab
        print(f"Calibrated {current_color}: BGR={tuple(avg_bgr.astype(int))}, Lab={tuple(avg_lab.astype(int))}")

def calibrate_colors(cap):
    global current_color, calibration_complete, calibrated_colors
    colors_to_calibrate = [color for color in Color]
    calibrated_colors = {}
    
    cv2.namedWindow('Calibration')
    
    for color in colors_to_calibrate:
        current_color = color
        while color not in calibrated_colors:
            ret, frame = cap.read()
            if not ret:
                return None
            # Provide instruction on the frame.
            cv2.putText(frame, f"Click on {color} color", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow('Calibration', frame)
            # Set the mouse callback with the current frame.
            cv2.setMouseCallback('Calibration', mouse_callback, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None
    
    cv2.destroyWindow('Calibration')
    return calibrated_colors

def get_cell_color(frame, point):
    """
    Sample color from a point in the frame and classify it as one of the Rubik's colors.
    This function converts the sampled ROI to the Lab color space.
    """
    x, y = int(point[0]), int(point[1])
    h, w = frame.shape[:2]
    # Ensure ROI is within image boundaries.
    if x - 2 < 0 or x + 3 > w or y - 2 < 0 or y + 3 > h:
        return None, (0, 0, 0)
    roi = frame[y-2:y+3, x-2:x+3]
    # Convert ROI to Lab
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    avg_lab = np.mean(roi_lab, axis=(0, 1))
    
    # Compare against calibrated Lab values.
    min_dist = float('inf')
    closest_color = None
    for color_name, color_lab in calibrated_colors.items():
        dist = np.linalg.norm(avg_lab - color_lab)
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
            
    # Also return the average Lab values (for later averaging).
    return closest_color, tuple(avg_lab.astype(int))

def transform_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """Transform 3D points using the marker pose and project them to 2D image coordinates."""
    # Convert rotation vector to rotation matrix.
    rot_matrix, _ = cv2.Rodrigues(rvec)
    points_transformed = np.dot(rot_matrix, points_3d.T).T + tvec
    points_2d, _ = cv2.projectPoints(points_transformed, np.zeros(3), np.zeros(3), 
                                     camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)

def process_frame(frame, detector, detected_faces=None):
    """Process a single frame to detect a Rubik's cube face and sample its colors."""
    if detected_faces is None:
        detected_faces = {}
        
    try:
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            marker_id = ids[0][0]  # Get the first detected marker's ID.
            
            # Only process markers with a valid mapping.
            if marker_id not in MARKER_TO_COLOR:
                cv2.putText(frame, f"Invalid marker ID: {marker_id}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return True, detected_faces

            # Make a clean copy for sampling.
            clean_frame = frame.copy()
            
            # Estimate pose.
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_size, camera_matrix, dist_coeffs)
            
            # Draw marker info on the display frame.
            draw_marker_info(frame, corners, ids, rvecs, tvecs)
            
            # Define the 3x3 grid positions (on the marker's plane, z=0)
            grid_positions = np.array([
                [1, -1, 0], [0, -1, 0], [-1, -1, 0],
                [1,  0, 0], [0,  0, 0], [-1,  0, 0],
                [1,  1, 0], [0,  1, 0], [-1,  1, 0]
            ]) * cell_size
            
            # Transform grid positions into image coordinates.
            sample_points = transform_points(grid_positions, 
                                          rvecs[0], tvecs[0], 
                                          camera_matrix, dist_coeffs)
            
            # Sample colors from each cell.
            current_colors = []
            current_lab_values = []
            for i, pt in enumerate(sample_points):
                if i == 4:  # For the center cell, we already know its color.
                    current_colors.append(MARKER_TO_COLOR[marker_id])
                    current_lab_values.append((0, 0, 0))  # dummy; center is predetermined.
                    continue
                
                color_name, avg_lab = get_cell_color(clean_frame, pt)
                if color_name is None:
                    continue  # Skip if ROI is out of bounds.
                current_colors.append(color_name)
                current_lab_values.append(avg_lab)
                pt_int = tuple(pt.astype(int))
                cv2.circle(frame, pt_int, 3, (0, 0, 255), -1)
                cv2.putText(frame, f"{color_name}", 
                          (pt_int[0], pt_int[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Begin (or continue) sampling for this face.
            if marker_id not in detected_faces:
                detected_faces[marker_id] = RubiksCubeFace(marker_id)
                detected_faces[marker_id].start_time = time.time()
            
            face = detected_faces[marker_id]
            current_time = time.time()
            
            if not face.is_sampling_complete():
                # Wait 1 second for the face to settle.
                if current_time - face.start_time >= 1.0:
                    samples_left = 5 - len(face.samples)
                    # Only sample if at least 500ms have passed since the last sample.
                    if not face.sampling_times or current_time - face.sampling_times[-1] >= 0.5:
                        face.add_sample(current_colors, current_lab_values)
                    cv2.putText(frame, f"Sampling face {marker_id}: {samples_left} samples left", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    time_left = 1.0 - (current_time - face.start_time)
                    cv2.putText(frame, f"Hold still face {marker_id}... {time_left:.1f}s", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if face.is_sampling_complete():
                    face.compute_final_colors()
                    print(f"\nFace {marker_id} colors:")
                    print("-------------")
                    for i in range(0, 9, 3):
                        print(f"{face.final_colors[i]} {face.final_colors[i+1]} {face.final_colors[i+2]}")
                    print("-------------")
            else:
                cv2.putText(frame, f"Face {marker_id} already captured", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show number of faces still needed.
        faces_left = 6 - len([f for f in detected_faces.values() if f.is_sampling_complete()])
        cv2.putText(frame, f"Faces left: {faces_left}", 
                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if faces_left == 0:
            cv2.putText(frame, "All faces captured!", 
                      (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Rubik Face Detection', frame)
        return cv2.waitKey(1) & 0xFF != ord('q'), detected_faces

    except Exception as e:
        print(f"Error processing frame: {e}")
        return False, detected_faces

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Perform color calibration.
    global calibrated_colors
    calibrated_colors = calibrate_colors(cap)
    if calibrated_colors is None:
        print("Calibration cancelled")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Prepare the ArUco detector.
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    detected_faces = {}
    cube_string_printed = False  # Make sure we print the cube string only once.
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        continue_processing, detected_faces = process_frame(frame, detector, detected_faces)
        if not continue_processing:
            break

        # Once all six faces have been captured, build and print the cube definition string.
        if (not cube_string_printed and len(detected_faces) == 6 and 
            all(face.is_sampling_complete() for face in detected_faces.values())):
            
            # Create a mapping from center color to the corresponding RubiksCubeFace.
            face_by_color = {}
            for face in detected_faces.values():
                face_by_color[face.center_color] = face

            try:
                # Build the cube definition string in the order:
                # Up (center color WHITE), Right (RED), Front (GREEN), Down (YELLOW), Left (ORANGE), Back (BLUE)
                cube_str = ''.join([
                    ''.join(color_to_letter[color] for color in face_by_color[Color.WHITE].final_colors),
                    ''.join(color_to_letter[color] for color in face_by_color[Color.RED].final_colors),
                    ''.join(color_to_letter[color] for color in face_by_color[Color.GREEN].final_colors),
                    ''.join(color_to_letter[color] for color in face_by_color[Color.YELLOW].final_colors),
                    ''.join(color_to_letter[color] for color in face_by_color[Color.ORANGE].final_colors),
                    ''.join(color_to_letter[color] for color in face_by_color[Color.BLUE].final_colors),
                ])
                print("\nCube definition string:")
                print(cube_str)
                cube_string_printed = True
            except KeyError as e:
                print("Error: Missing face", e)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
