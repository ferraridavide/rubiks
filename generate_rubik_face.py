import cv2
import numpy as np
import random
from generate_aruco import generate_aruco_marker

# Rubik's cube colors in BGR format
COLORS = {
    'white': (255, 255, 255),
    'yellow': (0, 255, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'orange': (0, 165, 255)
}

def create_colored_square(color, size=200):
    """Create a square of specified color"""
    return np.full((size, size, 3), color, dtype=np.uint8)

def generate_random_face(marker_id=1, square_size=200):
    """Generate a 3x3 Rubik's face with random colors and ArUco marker in center"""
    # Create 3x3 grid
    face = np.zeros((square_size * 3, square_size * 3, 3), dtype=np.uint8)
    
    # Get random colors
    colors = list(COLORS.values())
    random_colors = random.choices(colors, k=8)  # We need 8 colors (center will be marker)
    
    # Fill grid with random colors
    color_idx = 0
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:  # Center position
                # Generate ArUco marker
                marker = generate_aruco_marker(marker_id)
                # Convert grayscale marker to BGR
                square = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            else:
                square = create_colored_square(random_colors[color_idx])
                color_idx += 1
                
            # Place square in grid
            face[i*square_size:(i+1)*square_size, 
                 j*square_size:(j+1)*square_size] = square
    
    return face

def main():
    face = generate_random_face()
    cv2.imwrite('out/rubik_face_random.png', face)
    print("Generated random Rubik's face with ArUco marker as rubik_face_random.png")

if __name__ == "__main__":
    main()