from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time

app = Flask(__name__)

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize global variables
bpm = 0
rr = 0
temperature = 0

def estimate_vital_signs(frame):
    global bpm, rr, temperature
    # Placeholder for actual signal processing algorithms
    bpm = np.random.uniform(60, 100)  # Simulated HR value
    rr = np.random.uniform(12, 20)    # Simulated RR value
    temperature = np.random.uniform(36.0, 37.5)  # Simulated temperature value

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process frame to estimate vital signs
            estimate_vital_signs(frame)

            # Display estimated vital signs on the frame
            display_vital_signs(frame)

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Draw a green rectangle around each detected face
            for (x, y, w, h) in faces:
                # Draw the green box (ROI)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract the ROI (Region of Interest) of the detected face
                roi = frame[y:y + h, x:x + w]

                # Print the RGB values of the face region
                avg_rgb = np.mean(roi, axis=(0, 1))  # Calculate the average RGB values of the ROI
                print(f"Average RGB values of detected face: {avg_rgb}")

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Send the frame as a response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def display_vital_signs(frame):
    # Define positions for text
    positions = {
        'HR': (10, 50),
        'RR': (10, 100),
        'Temp': (10, 150)
    }
    # Define normal ranges
    normal_ranges = {
        'HR': (60, 100),
        'RR': (12, 20),
        'Temp': (36.0, 37.5)
    }
    # Define colors
    colors = {
        'normal': (0, 255, 0),  # Green
        'abnormal': (0, 0, 255)  # Red
    }
    # Prepare vital signs data
    vitals = {
        'HR': (bpm, 'bpm'),
        'RR': (rr, 'breaths/min'),
        'Temp': (temperature, 'Celsius')  # Changed from '°C' to 'ºC'
    }
    # Display each vital sign
    for key, (value, unit) in vitals.items():
        min_val, max_val = normal_ranges[key]
        color = colors['normal'] if min_val <= value <= max_val else colors['abnormal']
        text = f'{key}: {value:.2f} {unit}'
        position = positions[key]
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vital_signs')
def vital_signs():
    return jsonify({'bpm': bpm, 'rr': rr, 'temperature': temperature})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
