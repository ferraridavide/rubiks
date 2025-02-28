import numpy as np
import cv2
import time

def calibrate_camera():
    
    CHECKERBOARD = (6,9)  
    
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
    
    
    objpoints = []
    imgpoints = []
    
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None, None
        
    image_count = 0
    
    last_detection_time = 0
    
    while image_count < 15:  
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue
            
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                cv2.CALIB_CB_FAST_CHECK)
        
        
        if ret:
            
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners2, ret)
            
            
            cv2.putText(display_frame, "CHESSBOARD DETECTED - Press 's' to save", (20, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            last_detection_time = time.time()
        else:
            
            cv2.putText(display_frame, "NO CHESSBOARD DETECTED", (20, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        
        cv2.putText(display_frame, f"Images: {image_count}/15", (20, 80), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                  
        cv2.imshow('Calibration', display_frame)
        
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and ret:  
            objpoints.append(objp)
            imgpoints.append(corners2)
            image_count += 1
            print(f"Saved image {image_count}/15")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(objpoints) == 0:
        print("No calibration images captured!")
        return None, None
        
    print(f"Calibrating camera with {len(objpoints)} images...")
    
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                      gray.shape[::-1], None, None)
    
    print("Camera Matrix:")
    print(mtx)
    print("\nDistortion Coefficients:")
    print(dist)
    
    return mtx, dist

if __name__ == "__main__":
    camera_matrix, dist_coeffs = calibrate_camera()