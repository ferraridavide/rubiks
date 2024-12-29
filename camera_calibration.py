import numpy as np
import cv2

def calibrate_camera():
    # Chessboard dimensions
    CHECKERBOARD = (6,9)  # Adjust based on your printed chessboard
    
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
    
    # Arrays to store object points and image points
    objpoints = []
    imgpoints = []
    
    # Start capturing
    cap = cv2.VideoCapture(0)
    image_count = 0
    
    while image_count < 15:  # Capture 15 valid images
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        # If found, add object points, image points
        if ret:
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
            key = cv2.waitKey(1000)
            if key == ord('s'):  # Press 's' to save the image
                objpoints.append(objp)
                imgpoints.append(corners)
                image_count += 1
                print(f"Saved image {image_count}/15")
        
        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                      gray.shape[::-1], None, None)
    
    print("Camera Matrix:")
    print(mtx)
    print("\nDistortion Coefficients:")
    print(dist)
    
    return mtx, dist

if __name__ == "__main__":
    camera_matrix, dist_coeffs = calibrate_camera()