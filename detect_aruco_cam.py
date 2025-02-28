import cv2
import numpy as np


# Camera calibration parameters 
camera_matrix = np.array([[763.7055796,    0,         420.00145036],
                        [  0,         756.64528587, 202.20156225],
                        [  0,           0,           1        ]], dtype=float)
dist_coeffs = np.array([[ 0.09461912, -0.96563609, -0.00633544,  0.02877166,  2.01734678]], dtype=float)  # np.zeros((4, 1))

def draw_marker_info(frame, corners, ids, rvecs, tvecs):
    """Draw marker ID and pose information on frame"""
    if ids is not None:
        for i, corner in enumerate(corners):
            # Convert to int and reshape corner array
            corner = corner.reshape((4, 2)).astype(int)
            # Draw marker boundary
            cv2.polylines(frame, [corner], True, (0, 255, 0), 2)
            # Put marker ID
            center = corner.mean(axis=0).astype(int)
            cv2.putText(frame, f"id: {ids[i]}",
                        (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            # Draw axis if pose is available
            if rvecs is not None and tvecs is not None:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvecs[i], tvecs[i], 0.1)


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # ArUco dictionary setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break

        try:
            # Detect markers
            corners, ids, rejected = detector.detectMarkers(frame)

            # If markers detected, estimate pose
            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, 0.05, camera_matrix, dist_coeffs)

                # Draw detection results
                draw_marker_info(frame, corners, ids, rvecs, tvecs)

            # Display frame
            cv2.imshow('ArUco Detection', frame)

            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error processing frame: {e}")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
