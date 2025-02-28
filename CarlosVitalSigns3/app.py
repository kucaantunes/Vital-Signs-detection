from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize global variables for vital signs
bpm = 0
rr = 0
temperature = 0
blood_pressure_systolic = 120  # Placeholder value
blood_pressure_diastolic = 80  # Placeholder value
spo2 = 98  # Placeholder value (Oxygen Saturation in %)

# Store the RGB values for the detected face
avg_rgb_values = [0, 0, 0]

def estimate_vital_signs(frame):
    global bpm, rr, temperature, blood_pressure_systolic, blood_pressure_diastolic, spo2
    # Placeholder values for the vital signs (simulation)
    bpm = np.random.uniform(60, 100)  # Simulated Heart Rate in bpm
    rr = np.random.uniform(12, 20)    # Simulated Respiratory Rate in breaths per minute
    temperature = np.random.uniform(36.0, 37.5)  # Simulated Body Temperature in Celsius
    blood_pressure_systolic = np.random.randint(100, 130)  # Simulated Systolic BP in mmHg
    blood_pressure_diastolic = np.random.randint(60, 90)   # Simulated Diastolic BP in mmHg
    spo2 = np.random.uniform(95, 100)  # Simulated Oxygen Saturation in %

def gen_frames():
    global avg_rgb_values
    cap = cv2.VideoCapture(0)  # Initialize the webcam (0 for default webcam)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    # Set the frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)  # Set width to 1600 pixels for bigger video feed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)  # Set height to 600 pixels to reduce size

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            break
        else:
            # Process frame to estimate vital signs
            estimate_vital_signs(frame)

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around the faces and extract RGB values
            for (x, y, w, h) in faces:
                # Draw the green box (ROI)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract the ROI (Region of Interest) of the detected face
                roi = frame[y:y + h, x:x + w]

                # Calculate the average RGB values of the face region
                avg_rgb_values = np.mean(roi, axis=(0, 1))  # Calculate average RGB
                print(f"Average RGB values of detected face: {avg_rgb_values}")

            # Display the vital signs on the frame
            display_vital_signs(frame)

            # Encode the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Send the frame as part of a multipart response (MJPEG stream)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vital_signs')
def vital_signs():
    return jsonify({
        'bpm': bpm,
        'rr': rr,
        'temperature': temperature,
        'bp': f'{blood_pressure_systolic}/{blood_pressure_diastolic}',  # Combine systolic and diastolic
        'spo2': spo2,
        'rgb': avg_rgb_values.tolist()  # Convert numpy array to list for JSON serialization
    })

@app.route('/')
def index():
    return render_template('index.html')

def display_vital_signs(frame):
    global avg_rgb_values

    # Define positions for text
    positions = {
        'HR': (10, 50),
        'RR': (10, 100),
        'Temp': (10, 150),
        'BP': (10, 200),
        'SpO2': (10, 250),
        'RGB': (10, 300)
    }
    # Prepare vital signs data
    vitals = {
        'HR': (bpm, 'bpm'),
        'RR': (rr, 'breaths/min'),
        'Temp': (temperature, 'Celsius'),  # Changed from 'ºC' to '°C'
        'BP': (f'{blood_pressure_systolic}/{blood_pressure_diastolic}', 'mmHg'),
        'SpO2': (f'{spo2:.2f}', '%')
    }

    # Display each vital sign
    for key, (value, unit) in vitals.items():
        text = f'{key}: {value} {unit}'
        position = positions[key]
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display RGB values on the frame
    rgb_text = f"RGB: R-{int(avg_rgb_values[2])} G-{int(avg_rgb_values[1])} B-{int(avg_rgb_values[0])}"
    cv2.putText(frame, rgb_text, positions['RGB'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

if __name__ == "__main__":
    app.run(debug=True)
