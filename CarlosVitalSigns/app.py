from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time

app = Flask(__name__)

# Fake vital sign generation (for demonstration purposes)
def get_vital_signs():
    bpm = np.random.uniform(55, 105)
    systolic_bp = np.random.uniform(75, 130)
    temp = np.random.uniform(35.5, 38.0)
    spo2 = np.random.uniform(90, 100)
    breathing = np.random.uniform(10, 25)
    return {
        "bpm": round(bpm, 2),
        "bp": round(systolic_bp, 2),
        "temp": round(temp, 2),
        "spo2": round(spo2, 2),
        "breathing": round(breathing, 2)
    }

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Draw rectangle around the face (ROI)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Send the frame to the web
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vital_signs')
def vital_signs():
    return jsonify(get_vital_signs())

if __name__ == "__main__":
    app.run(debug=True)
