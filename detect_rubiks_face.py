import cv2
import numpy as np
from detect_aruco_cam import camera_matrix, dist_coeffs, draw_marker_info

def get_cell_color(frame, point):
    """Sample color from a point and classify it as a Rubik's color"""
    # BGR colors for Rubik's cube
    COLORS = {
        'white': (255, 255, 255),
        'yellow': (0, 255, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'orange': (0, 165, 255)
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
            # Create visualization frame
            vis_frame = frame.copy()
            
            # Detect markers
            corners, ids, rejected = detector.detectMarkers(frame)

            if ids is not None:
                # Get pose
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, 0.05, camera_matrix, dist_coeffs)
                
                # Draw marker info on visualization frame only
                draw_marker_info(vis_frame, corners, ids, rvecs, tvecs)
                
                # Get marker corners
                marker_corners = corners[0][0]
                marker_center = np.mean(marker_corners, axis=0)
                
                # Estimate cell size from marker size
                cell_size = np.linalg.norm(marker_corners[0] - marker_corners[1]) * 1.1
                
                # Sample colors from surrounding cells
                grid_positions = [
                    (-1, -1), (0, -1), (1, -1),
                    (-1,  0),          (1,  0),
                    (-1,  1), (0,  1), (1,  1)
                ]
                
                for dx, dy in grid_positions:
                    # Calculate sampling point
                    sample_point = marker_center + np.array([dx, dy]) * cell_size
                    
                    # Get color from original frame but draw on visualization frame
                    color_name, avg_color = get_cell_color(frame, sample_point)
                    
                    # Draw sampling point and color name on visualization frame
                    cv2.circle(vis_frame, tuple(sample_point.astype(int)), 3, (0,0,255), -1)
                    cv2.putText(vis_frame, color_name, 
                              (int(sample_point[0]), int(sample_point[1] - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            cv2.imshow('Rubik Face Detection', vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error processing frame: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()