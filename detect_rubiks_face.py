import cv2
import numpy as np
import argparse
from detect_aruco_cam import camera_matrix, dist_coeffs, draw_marker_info

# Global variables for calibration
calibrated_colors = {}
current_color = None
calibration_complete = False

def mouse_callback(event, x, y, flags, param):
    global calibrated_colors, current_color, calibration_complete
    if event == cv2.EVENT_LBUTTONDOWN and current_color is not None:
        frame = param
        # Sample color from 5x5 region
        roi = frame[y-2:y+3, x-2:x+3]
        avg_color = tuple(np.mean(roi, axis=(0, 1)).astype(int))
        calibrated_colors[current_color] = avg_color
        print(f"Calibrated {current_color}: {avg_color}")

def calibrate_colors(cap):
    global current_color, calibration_complete, calibrated_colors
    colors_to_calibrate = ['white', 'yellow', 'blue', 'green', 'red', 'orange']
    calibrated_colors = {}
    
    cv2.namedWindow('Calibration')
    
    for color in colors_to_calibrate:
        current_color = color
        while color not in calibrated_colors:
            ret, frame = cap.read()
            if not ret:
                return None
                
            cv2.putText(frame, f"Click on {color} color", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow('Calibration', frame)
            cv2.setMouseCallback('Calibration', mouse_callback, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None
    
    cv2.destroyWindow('Calibration')
    return calibrated_colors

def get_cell_color(frame, point):
    """Sample color from a point and classify it as a Rubik's color"""
    global calibrated_colors
    
    # Sample color from 5x5 region around point
    x, y = int(point[0]), int(point[1])
    roi = frame[y-2:y+3, x-2:x+3]
    avg_color = np.mean(roi, axis=(0, 1))
    
    # Find closest calibrated color
    min_dist = float('inf')
    closest_color = None
    for color_name, color_bgr in calibrated_colors.items():
        dist = np.linalg.norm(avg_color - color_bgr)
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
            
    return closest_color, tuple(avg_color.astype(int))

def transform_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """Transform 3D points using rotation and translation, then project to 2D"""
    # Convert rotation vector to rotation matrix
    rot_matrix, _ = cv2.Rodrigues(rvec)
    
    # Transform points
    points_transformed = np.dot(rot_matrix, points_3d.T).T + tvec
    
    # Project to image plane
    points_2d, _ = cv2.projectPoints(points_transformed, np.zeros(3), np.zeros(3), 
                                   camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)

def process_frame(frame, detector):
    """Process a single frame to detect Rubik's cube face"""
    try:
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            # Create clean copy for sampling
            clean_frame = frame.copy()
            
            # Get pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.05, camera_matrix, dist_coeffs)
            
            # Draw marker info on display frame only
            draw_marker_info(frame, corners, ids, rvecs, tvecs)
            
            # Define grid positions in 3D space (z=0 plane)
            marker_size = 0.05  # 5cm marker
            cell_size = marker_size * 1.5
            grid_positions = np.array([
                [-1, -1, 0], [0, -1, 0], [1, -1, 0],
                [-1,  0, 0],             [1,  0, 0],
                [-1,  1, 0], [0,  1, 0], [1,  1, 0]
            ]) * cell_size
            
            # Transform grid positions using marker pose
            sample_points = transform_points(grid_positions, 
                                          rvecs[0], tvecs[0], 
                                          camera_matrix, dist_coeffs)
            
            # Sample colors from transformed points
            for pt in sample_points:
                color_name, avg_color = get_cell_color(clean_frame, pt)
                pt_int = tuple(pt.astype(int))
                cv2.circle(frame, pt_int, 3, (0,0,255), -1)
                cv2.putText(frame, f"{color_name}", 
                          (pt_int[0], pt_int[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        cv2.imshow('Rubik Face Detection', frame)
        return cv2.waitKey(1) & 0xFF != ord('q')

    except Exception as e:
        print(f"Error processing frame: {e}")
        return False

def load_from_image(image_path):
    """Process a single image file"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Cannot load image: {image_path}")
        return
    
    # ArUco setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    process_frame(frame, detector)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image = None
    if image:
        load_from_image(image)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Add calibration step
    global calibrated_colors
    calibrated_colors = calibrate_colors(cap)
    if calibrated_colors is None:
        print("Calibration cancelled")
        cap.release()
        cv2.destroyAllWindows()
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret or not process_frame(frame, detector):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()