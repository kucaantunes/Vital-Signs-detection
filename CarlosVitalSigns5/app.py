import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
import time
from scipy.signal import find_peaks
import mediapipe as mp
from transformers import BartForConditionalGeneration, BartTokenizer  # Updated import for BART

app = Flask(__name__)

# Initialize global variables for vital signs
bpm = 0
temperature = 36.5
blood_pressure_systolic = 120
blood_pressure_diastolic = 80
spo2 = 98
respiratory_rate = 16
green_intensity_values = []
red_intensity_values = []
respiratory_intensity_values = []

# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# FaceMesh and Region of Interest (ROI) for heartbeat calculation
class ROIDetector:
    _lower_face = [200, 431, 411, 340, 349, 120, 111, 187, 211]  # Mesh indices for lower face

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame):
        """Find single face in frame and extract lower half of the face."""
        results = self.face_mesh.process(frame)
        point_list = []
        if results.multi_face_landmarks is not None:
            coords = self.get_facemesh_coords(results.multi_face_landmarks[0], frame)
            point_list = coords[self._lower_face, :2]  # :2 -> only x and y
        roimask = self.fill_roimask(point_list, frame)
        return roimask, results

    def get_facemesh_coords(self, landmark_list, img):
        """Extract FaceMesh landmark coordinates into NumPy array."""
        h, w = img.shape[:2]  # grab width and height from image
        xyz = [(lm.x, lm.y, lm.z) for lm in landmark_list.landmark]
        return np.multiply(xyz, [w, h, w]).astype(int)

    def fill_roimask(self, point_list, img):
        """Create binary mask, filled inside contour given by list of points."""
        mask = np.zeros(img.shape[:2], dtype="uint8")
        if len(point_list) > 2:
            contours = np.reshape(point_list, (1, -1, 1, 2))  # Expected by OpenCV
            cv2.drawContours(mask, contours, 0, color=255, thickness=cv2.FILLED)
        return mask

# Temperature related functions (from the provided code)
TEMP_TUNER = 2.25
TEMP_TOLERENCE = 70.6

def convert_to_temperature(pixel_avg):
    """
    Converts pixel value (mean) to temperature depending upon the camera hardware
    """
    f = pixel_avg / TEMP_TUNER
    c = (f - 32) * 5/9
    
    return c

def process_temperature(frame):
    frame = ~frame
    heatmap = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
    
    heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    ret, binary_thresh = cv2.threshold(heatmap_gray, 200, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    image_opening = cv2.dilate(image_erosion, kernel, iterations=1)
    
    # Get contours from the image obtained by opening operation
    contours, _ = cv2.findContours(image_opening, 1, 2)

    image_with_rectangles = np.copy(heatmap)

    for contour in contours:
        # rectangle over each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        if (w) * (h) < 2400:
            continue

        # Mask is boolean type of matrix.
        mask = np.zeros_like(heatmap_gray)
        cv2.drawContours(mask, contour, -1, 255, -1)

        # Mean of only those pixels which are in blocks and not the whole rectangle selected
        mean = convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])

        # Colors for rectangles and textmin_area
        temperature = round(mean, 2)
        color = (0, 255, 0) if temperature < TEMP_TOLERENCE else (
            255, 255, 127)
        
        # Draw rectangles for visualisation
        image_with_rectangles = cv2.rectangle(image_with_rectangles, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_with_rectangles, "{} C".format(temperature), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    return temperature

# Heart rate estimation function using the webcam's green channel
def estimate_heart_rate_via_camera(frame):
    global green_intensity_values
    rawimg = frame.copy()
    roid = ROIDetector()
    roimask, results = roid.process(frame)
    g, _, _, _ = cv2.mean(rawimg, mask=roimask)  # Get green intensity
    green_intensity_values.append(g)

    if len(green_intensity_values) > 300:
        green_intensity_values.pop(0)

    if len(green_intensity_values) > 1:
        signal = np.array(green_intensity_values)
        signal -= np.mean(signal)  # Remove DC component

        # Use FFT to detect the heart rate
        fft_result = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(len(signal), d=1/30)  # Assuming 30 fps
        dominant_freq = fft_freqs[np.argmax(np.abs(fft_result))]

        heart_rate = dominant_freq * 60  # Convert to BPM
        return heart_rate
    return 0

# Video stream processing
def gen_frames():
    global bpm, temperature, blood_pressure_systolic, blood_pressure_diastolic, spo2, respiratory_rate

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bpm = estimate_heart_rate_via_camera(frame)
        respiratory_rate = estimate_respiratory_rate_via_motion(frame)
        spo2 = estimate_spo2()
        temperature = process_temperature(frame)  # Added temperature processing
        blood_pressure_systolic, blood_pressure_diastolic = estimate_blood_pressure()

        # Display vital signs on the frame
        cv2.putText(frame, f"Heart Rate: {bpm:.1f} bpm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Respir. Rate: {respiratory_rate:.1f} bpm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"SpO2: {spo2:.1f} %", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, f"Temp: {temperature:.1f} °C", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"BP: {blood_pressure_systolic:.1f}/{blood_pressure_diastolic:.1f} mmHg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vital_signs')
def vital_signs():
    return jsonify({
        'bpm': bpm,
        'temperature': temperature,
        'bp': f'{blood_pressure_systolic}/{blood_pressure_diastolic}',
        'spo2': spo2,
        'rr': respiratory_rate,
    })

@app.route('/bart_report')
def bart_report():
    report = generate_bart_report()
    return jsonify({'report': report})

@app.route('/gpt2_report')
def gpt2_report():
    report = generate_gpt2_report()
    return jsonify({'report': report})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
