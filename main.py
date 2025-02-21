import pickle
import os

import numpy as np
from camera_calibration import calibrate_camera
import detect_rubiks_face
from visualization.cube3d import start
import importlib
import sys

def main():
    # Load or create camera calibration
    CALIB_FILE = 'camera_calibration.pkl'
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, 'rb') as f:
            camera_matrix, dist_coeffs = pickle.load(f)
    else:
        camera_matrix, dist_coeffs = calibrate_camera()
        with open(CALIB_FILE, 'wb') as f:
            pickle.dump((camera_matrix, dist_coeffs), f)

    # Start cube detection
    detect_rubiks_face.camera_matrix = camera_matrix
    detect_rubiks_face.dist_coeffs = dist_coeffs
    
    print("Starting Rubik's cube face detection...")
    print("Please show each face of the cube to the camera.")
    print("Make sure to hold the cube steady while capturing.")
    
    # Run the detection and get the cube configuration
    cube_config = detect_rubiks_face.main()
    
    if cube_config:
        start(cube_config)
    else:
        print("Failed to capture cube configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()
