from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import threading
import queue
import time
from scipy.fft import fft
import json

app = Flask(__name__)

# Initialize AI models
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use GPT-2 for text generation
llm_model = GPT2LMHeadModel.from_pretrained("gpt2")
llm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
llm_model.to(device)
# Use BLIP for image analysis
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
vlm_model.to(device)

# Global variables for vital signs
vital_signs = {
    'temperature': 0,
    'heart_rate': 0,
    'blood_pressure_sys': 0,
    'blood_pressure_dia': 0,
    'breath_rate': 0,
    'spo2': 0
}

# Signal processing parameters
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
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

class VitalSignsProcessor:
    def __init__(self):
        self.frame_buffer = []
        self.roi_buffer = queue.Queue(maxsize=WINDOW_SIZE)
        
    def extract_rois(self, frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            forehead_roi = frame[y:y+int(h*0.3), x+int(w*0.3):x+int(w*0.7)]
            cheek_roi = frame[y+int(h*0.3):y+int(h*0.7), x+int(w*0.3):x+int(w*0.7)]
            return forehead_roi, cheek_roi, (x, y, w, h)
        return None, None, None

    def calculate_vital_signs(self, forehead_roi, cheek_roi):
        if forehead_roi is None or cheek_roi is None:
            return
            
        # Extract RGB values
        forehead_green = np.mean(forehead_roi[:, :, 1])
        cheek_green = np.mean(cheek_roi[:, :, 1])
        
        # Update buffer
        if self.roi_buffer.full():
            self.roi_buffer.get()
        self.roi_buffer.put((forehead_green, cheek_green))
        
        # Process signals when buffer is full
        if self.roi_buffer.qsize() == WINDOW_SIZE:
            signals = np.array(list(self.roi_buffer.queue))
            forehead_signal = process_ppg_signal(signals[:, 0])
            cheek_signal = process_ppg_signal(signals[:, 1])
            
            # Calculate heart rate
            fft_result = np.abs(fft(forehead_signal))
            freq = np.fft.fftfreq(len(forehead_signal), 1/SAMPLE_RATE)
            peak_freq = freq[np.argmax(fft_result[1:]) + 1]
            heart_rate = peak_freq * 60
            
            # Calculate SpO2 (simplified)
            r_ratio = np.std(forehead_signal) / np.std(cheek_signal)
            spo2 = 110 - 25 * r_ratio
            
            # Update vital signs
            vital_signs.update({
                'temperature': 36.5 + np.random.normal(0, 0.1),  # Simulated
                'heart_rate': heart_rate,
                'blood_pressure_sys': 120 + heart_rate/10,  # Estimated
                'blood_pressure_dia': 80 + heart_rate/15,   # Estimated
                'breath_rate': heart_rate/4,  # Estimated
                'spo2': min(100, max(90, spo2))
            })

def generate_frames():
    camera = cv2.VideoCapture(0)
    processor = VitalSignsProcessor()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Process vital signs
        forehead_roi, cheek_roi, face_coords = processor.extract_rois(frame)
        if forehead_roi is not None:
            processor.calculate_vital_signs(forehead_roi, cheek_roi)
            
            # Draw ROIs and values on frame
            x, y, w, h = face_coords
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x+int(w*0.3), y), (x+int(w*0.7), y+int(h*0.3)), (255, 0, 0), 2)  # forehead
            cv2.rectangle(frame, (x+int(w*0.3), y+int(h*0.3)), (x+int(w*0.7), y+int(h*0.7)), (0, 0, 255), 2)  # cheek
            
            cv2.putText(frame, f"HR: {vital_signs['heart_rate']:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"SpO2: {vital_signs['spo2']:.1f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vital_signs')
def get_vital_signs():
    return jsonify(vital_signs)

@app.route('/analyze_health')
def analyze_health():
    prompt = f"""Analyzing vital signs:
    Temperature: {vital_signs['temperature']}°C
    Heart Rate: {vital_signs['heart_rate']} bpm
    Blood Pressure: {vital_signs['blood_pressure_sys']}/{vital_signs['blood_pressure_dia']} mmHg
    Breath Rate: {vital_signs['breath_rate']} breaths/min
    SpO2: {vital_signs['spo2']}%
    
    Based on these values, provide a health analysis regarding COVID-19 and pneumonia risks:"""
    
    # Generate LLM analysis
    inputs = llm_tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(
        inputs,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    analysis = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'analysis': analysis})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    prompt = f"User question about vital signs: {user_message}\nCurrent vital signs: {json.dumps(vital_signs)}\nMedical Assistant:"
    
    inputs = llm_tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7
    )
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)