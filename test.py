import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

# Add these constants at the top with other imports
REQUIRED_COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'white']
SAMPLES_PER_COLOR = 3

class ColorCalibrator:
    def __init__(self):
        self.color_samples = defaultdict(list)
        self.calibrated_ranges = {}
        self.is_calibrated = False
        self.current_color_index = 0
        self.current_sample_count = 0
    
    def get_next_color_prompt(self):
        if self.current_color_index >= len(REQUIRED_COLORS):
            return None
        remaining = SAMPLES_PER_COLOR - self.current_sample_count
        color = REQUIRED_COLORS[self.current_color_index]
        return f"Click on {color} ({remaining} samples needed)"
    
    def add_sample(self, hsv_value):
        current_color = REQUIRED_COLORS[self.current_color_index]
        self.color_samples[current_color].append(hsv_value)
        self.current_sample_count += 1
        
        if self.current_sample_count >= SAMPLES_PER_COLOR:
            self.current_color_index += 1
            self.current_sample_count = 0
            
        self.is_calibrated = self.current_color_index >= len(REQUIRED_COLORS)
        if self.is_calibrated:
            self.calibrate()
            print("Calibration completed!")
        
        return self.get_next_color_prompt()
        
    def calibrate(self):
        for color, samples in self.color_samples.items():
            if len(samples) < 3:  # Minimum samples needed
                continue
                
            samples = np.array(samples)
            h_values = samples[:, 0]
            s_values = samples[:, 1]
            v_values = samples[:, 2]
            
            # Calculate ranges with some margin
            h_margin = 5
            s_margin = 20
            v_margin = 20
            
            # Handle red's wrap-around case specially
            if color == 'red':
                if np.any(h_values > 170) and np.any(h_values < 10):
                    self.calibrated_ranges[color] = [
                        [0, np.max([0, np.min(s_values) - s_margin]), np.max([0, np.min(v_values) - v_margin])],
                        [10, np.min([255, np.max(s_values) + s_margin]), np.min([255, np.max(v_values) + v_margin])],
                        [170, np.max([0, np.min(s_values) - s_margin]), np.max([0, np.min(v_values) - v_margin])],
                        [180, np.min([255, np.max(s_values) + s_margin]), np.min([255, np.max(v_values) + v_margin])]
                    ]
                    continue
            
            # For other colors
            self.calibrated_ranges[color] = [
                [np.max([0, np.min(h_values) - h_margin]),
                 np.max([0, np.min(s_values) - s_margin]),
                 np.max([0, np.min(v_values) - v_margin])],
                [np.min([180, np.max(h_values) + h_margin]),
                 np.min([255, np.max(s_values) + s_margin]),
                 np.min([255, np.max(v_values) + v_margin])]
            ]
        
        self.is_calibrated = True

def apply_white_balance(frame):
    # Simple gray world white balance
    b, g, r = cv2.split(frame)
    b_avg = cv2.mean(b)[0]
    g_avg = cv2.mean(g)[0]
    r_avg = cv2.mean(r)[0]
    
    # Find the gain of each channel
    k = (b_avg + g_avg + r_avg) / 3
    kb = k / b_avg
    kg = k / g_avg
    kr = k / r_avg
    
    b = cv2.addWeighted(b, kb, 0, 0, 0)
    g = cv2.addWeighted(g, kg, 0, 0, 0)
    r = cv2.addWeighted(r, kr, 0, 0, 0)
    
    return cv2.merge([b, g, r])

def preprocess_image(frame):
    # White balance
    balanced = apply_white_balance(frame)
    
    # Enhance contrast using CLAHE in LAB space
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l,a,b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(enhanced)
    
    # Convert to HSV
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    return hsv, enhanced

def get_average_color(hsv_frame, x, y, window_size=5):
    # Get average color from a small window around the clicked point
    half_size = window_size // 2
    window = hsv_frame[max(0, y-half_size):min(hsv_frame.shape[0], y+half_size+1),
                      max(0, x-half_size):min(hsv_frame.shape[1], x+half_size+1)]
    return np.mean(window, axis=(0, 1))

def get_color_confidence(hsv_values, color_range):
    h, s, v = hsv_values
    
    if len(color_range) == 4:  # Red case with wrap-around
        conf1 = calculate_confidence(hsv_values, color_range[:2])
        conf2 = calculate_confidence(hsv_values, color_range[2:])
        return max(conf1, conf2)
    else:
        return calculate_confidence(hsv_values, color_range)

def calculate_confidence(hsv_values, range_pair):
    lower, upper = range_pair
    h, s, v = hsv_values
    
    # Calculate how centered the values are in their ranges
    h_conf = 1.0 - min(abs(h - (lower[0] + upper[0])/2) / ((upper[0] - lower[0])/2), 1.0)
    s_conf = 1.0 - min(abs(s - (lower[1] + upper[1])/2) / ((upper[1] - lower[1])/2), 1.0)
    v_conf = 1.0 - min(abs(v - (lower[2] + upper[2])/2) / ((upper[2] - lower[2])/2), 1.0)
    
    return (h_conf * 0.5 + s_conf * 0.3 + v_conf * 0.2)  # Weighted average

def get_dominant_colors(hsv_frame, x, y, window_size=15):
    half_size = window_size // 2
    window = hsv_frame[max(0, y-half_size):min(hsv_frame.shape[0], y+half_size+1),
                      max(0, x-half_size):min(hsv_frame.shape[1], x+half_size+1)]
    
    pixels = window.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(pixels)
    
    # Get the dominant color (most common cluster)
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_idx = unique[np.argmax(counts)]
    return kmeans.cluster_centers_[dominant_idx]

# Initialize calibrator
calibrator = ColorCalibrator()
calibration_mode = False

def mouse_callback(event, x, y, flags, param):
    global calibration_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = param['hsv_frame']
        dominant_color = get_dominant_colors(hsv_frame, x, y)
        
        if calibration_mode:
            next_prompt = calibrator.add_sample(dominant_color)
            if next_prompt is None:
                calibration_mode = False
                print("Exiting calibration mode")
            else:
                print(next_prompt)
        else:
            if not calibrator.is_calibrated:
                print("Please calibrate first (press 'c')")
                return
                
            # Find best matching color
            best_color = 'unknown'
            best_confidence = 0
            
            for color, ranges in calibrator.calibrated_ranges.items():
                confidence = get_color_confidence(dominant_color, ranges)
                if confidence > best_confidence and confidence > 0.5:
                    best_confidence = confidence
                    best_color = color
            
            print(f"Detected color: {best_color} (confidence: {best_confidence:.2f})")

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('Webcam Feed')
param_dict = {'hsv_frame': None}
cv2.setMouseCallback('Webcam Feed', mouse_callback, param_dict)

# Modify main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv_frame, processed = preprocess_image(frame)
    param_dict['hsv_frame'] = hsv_frame
    param_dict['frame'] = frame
    
    # Display calibration status and prompt
    if calibration_mode:
        prompt = calibrator.get_next_color_prompt()
        if prompt:
            cv2.putText(frame, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "DETECTION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Webcam Feed', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        if not calibrator.is_calibrated:
            calibration_mode = True
            calibrator = ColorCalibrator()  # Reset calibrator
            print(calibrator.get_next_color_prompt())
        else:
            print("Already calibrated!")

cap.release()
cv2.destroyAllWindows()
