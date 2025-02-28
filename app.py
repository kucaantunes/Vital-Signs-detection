from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import queue

app = Flask(__name__)

SAMPLE_RATE = 30
WINDOW_SIZE = 300
butter_order = 4
cutoff_freq = [0.7, 4.0]
nyq = SAMPLE_RATE / 2


def butter_bandpass():
    normal_cutoff = [x / nyq for x in cutoff_freq]
    b, a = butter(butter_order, normal_cutoff, btype='band')
    return b, a


def process_ppg_signal(signal):
    b, a = butter_bandpass()
    return filtfilt(b, a, signal)


class VitalSignsProcessor:
    def __init__(self):
        self.roi_buffer = queue.Queue(maxsize=WINDOW_SIZE)

    def extract_rois(self, frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            forehead_roi = frame[y:y + int(h * 0.2), x + int(w * 0.4):x + int(w * 0.6)]
            return forehead_roi, (x, y, w, h)
        return None, None

    def calculate_vital_signs(self, roi):
        if roi is None:
            return None

        roi_mean = np.mean(roi, axis=(0, 1))  # RGB mean
        rgb_values = {"R": roi_mean[2], "G": roi_mean[1], "B": roi_mean[0]}

        if self.roi_buffer.full():
            self.roi_buffer.get()
        self.roi_buffer.put(rgb_values["G"])

        if self.roi_buffer.qsize() == WINDOW_SIZE:
            signal = np.array(list(self.roi_buffer.queue))
            filtered_signal = process_ppg_signal(signal)
            fft_result = np.abs(fft(filtered_signal))
            freq = np.fft.fftfreq(len(filtered_signal), 1 / SAMPLE_RATE)
            peak_freq = freq[np.argmax(fft_result[1:]) + 1]
            heart_rate = peak_freq * 60

            temperature = 36.5 + (rgb_values["R"] - rgb_values["G"]) / 100
            spo2 = 98 - (rgb_values["B"] / 255) * 2
            breath_rate = heart_rate / 4
            blood_pressure_sys = 120 + heart_rate / 10
            blood_pressure_dia = 80 + heart_rate / 15

            return {
                "heart_rate": heart_rate,
                "temperature": temperature,
                "spo2": spo2,
                "breath_rate": breath_rate,
                "blood_pressure_sys": blood_pressure_sys,
                "blood_pressure_dia": blood_pressure_dia,
                "rgb": rgb_values
            }

        return None


def generate_frames():
    camera = cv2.VideoCapture(0)
    processor = VitalSignsProcessor()

    while True:
        success, frame = camera.read()
        if not success:
            break

        roi, face_coords = processor.extract_rois(frame)
        vitals = processor.calculate_vital_signs(roi)

        if vitals and face_coords:
            x, y, w, h = face_coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, f"HR: {vitals['heart_rate']:.1f} bpm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Temp: {vitals['temperature']:.1f}°C", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"SpO2: {vitals['spo2']:.1f}%", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Breath Rate: {vitals['breath_rate']:.1f} bpm", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"BP: {vitals['blood_pressure_sys']:.1f}/{vitals['blood_pressure_dia']:.1f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"RGB: R={vitals['rgb']['R']:.0f} G={vitals['rgb']['G']:.0f} B={vitals['rgb']['B']:.0f}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
