# Rubik's Cube Solver with Aruco Markers
<p align="center" width="100%">
    <img align="center" src="https://github.com/user-attachments/assets/fafcec40-334c-4039-98fb-d3a55b3c3c7f" width="50%">
</p>

This project uses computer vision to detect the state of a Rubik's cube and then provides a solution using the Kociemba algorithm (two-phase solver). It leverages ArUco markers placed on the center pieces of each face to identify the faces and their orientation.

## Features

* **Camera Calibration:** Calibrates your webcam to minimize distortion and improve accuracy.
* **Aruco Marker Detection:** Detects ArUco markers on the Rubik's Cube faces.
* **Color Detection:** Determines the color of each cubie on the detected face using a calibration process.
* **Cube State String Generation:** Creates the cube definition string needed by the Kociemba solver.
* **Solution Visualization:** Displays an interactive 3D Rubik's cube and animates the solution steps.

## Requirements

* Python 3.7+
* Libraries:
    * `numpy==2.2.1`
    * `opencv-contrib-python==4.10.0.84`
    * `kociemba==1.2.1`
    * `ursina` (for visualization)
