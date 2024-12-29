import cv2
import numpy as np
from detect_aruco_cam import camera_matrix, dist_coeffs, draw_marker_info

def get_cell_color(frame, point):
    """Sample color from a point and classify it as a Rubik's color"""
    # BGR colors for Rubik's cube
    COLORS = {
        'white': (227, 200, 186),
        'yellow': (215, 206, 215),
        'blue': (255, 0, 0),
        'green': (120, 227, 111),
        'red': (0, 0, 255),
        'orange': (109, 140, 220)
    }
    
    # Sample color from 5x5 region around point
    x, y = int(point[0]), int(point[1])
    roi = frame[y-2:y+3, x-2:x+3]
    avg_color = np.mean(roi, axis=(0, 1))
    
    # Find closest Rubik's color
    min_dist = float('inf')
    closest_color = None
    for color_name, color_bgr in COLORS.items():
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

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # ArUco setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
                cell_size = marker_size * 1.1
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
                    # Get color info using clean frame
                    color_name, avg_color = get_cell_color(clean_frame, pt)
                    
                    # Draw visualization on display frame
                    pt_int = tuple(pt.astype(int))
                    cv2.circle(frame, pt_int, 3, (0,0,255), -1)
                    cv2.putText(frame, f"{color_name}", 
                              (pt_int[0], pt_int[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            cv2.imshow('Rubik Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error processing frame: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()