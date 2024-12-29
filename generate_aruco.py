import cv2
import numpy as np

def generate_aruco_marker(marker_id, size=200, border_size=20):
    """Generate ArUco marker for given ID with white border"""
    # Calculate marker size without border
    marker_size = size - (2 * border_size)
    
    # Generate base marker
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    # Add white border
    # Cosa che ho trovato: aggiungere un bordo bianco al marker aiuta molto il riconoscimento, probabilmente perchè
    # il detector cerca i bordi del marker e avere contorno bianco aumenta il contrasto
    marker_with_border = np.pad(marker_image, pad_width=border_size, mode='constant', constant_values=255)
    
    return marker_with_border

def main():
    for idx in range(6):
        try:
            # Generate marker
            marker = generate_aruco_marker(idx)
            
            # Save marker
            filename = f'out/aruco_marker_{idx}.png'
            cv2.imwrite(filename, marker)
            print(f"Generated marker {idx} as {filename}")
            
        except Exception as e:
            print(f"Error generating marker {idx}: {e}")

if __name__ == "__main__":
    main()