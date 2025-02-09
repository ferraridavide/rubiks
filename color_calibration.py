import cv2
import numpy as np

# Load an image from file
def load_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not load image.")
        return None
    return frame

# Function to pick color on mouse click
def pick_color(event, x, y, flags, param):
    global hsv_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_value = hsv_frame[y, x]
        print(f"Picked HSV Color: {hsv_value}")

# Function to create trackbars for color tuning
def nothing(x):
    pass

def create_trackbars():
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("H Min", "Trackbars", 0, 180, nothing)
    cv2.createTrackbar("H Max", "Trackbars", 180, 180, nothing)
    cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

def get_trackbar_values():
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")
    return np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])

# Main function to process image and detect colors
def calibrate_colors(image_path):
    global hsv_frame
    
    frame = load_image(image_path)
    if frame is None:
        return
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    create_trackbars()
    
    while True:
        lower_bound, upper_bound = get_trackbar_values()
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Filtered Result", result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
    
    print(f"Final HSV Ranges: Lower-{lower_bound}, Upper-{upper_bound}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "image2.jpg"  # Change this to the path of your image
    calibrate_colors(image_path)
