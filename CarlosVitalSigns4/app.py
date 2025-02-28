from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer, GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Initialize global variables for vital signs
bpm = 0
temperature = 0
blood_pressure_systolic = 120  # Placeholder value
blood_pressure_diastolic = 80  # Placeholder value
spo2 = 98  # Placeholder value (Oxygen Saturation in %)
respiratory_rate = 16  # Placeholder value
avg_rgb_values = [0, 0, 0]  # To store the RGB values of the face region

# Load pre-trained models and tokenizers
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Initialize face cascade (Haar cascade for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def estimate_heart_rate(frame):
    bpm = np.random.uniform(60, 100)  # Simulated heart rate estimation
    return bpm

def estimate_temperature(frame):
    temperature = np.mean(frame)  # Simulated temperature estimation
    return temperature

def estimate_vital_signs(frame):
    global bpm, temperature, blood_pressure_systolic, blood_pressure_diastolic, spo2, respiratory_rate
    bpm = estimate_heart_rate(frame)  # Estimate heart rate
    temperature = estimate_temperature(frame)  # Estimate body temperature
    blood_pressure_systolic = np.random.randint(100, 130)  # Simulated Systolic BP in mmHg
    blood_pressure_diastolic = np.random.randint(60, 90)   # Simulated Diastolic BP in mmHg
    spo2 = np.random.uniform(95, 100)  # Simulated Oxygen Saturation in %
    respiratory_rate = np.random.randint(12, 20)  # Simulated Respiratory Rate

def gen_frames():
    global avg_rgb_values
    cap = cv2.VideoCapture(0)  # Open default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            break
        else:
            estimate_vital_signs(frame)

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi = frame[y:y + h, x:x + w]  # Region of interest: the detected face
                avg_rgb_values = np.mean(roi, axis=(0, 1))  # Calculate average RGB values of the face area

            # Send the frame as part of a multipart response (MJPEG stream)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

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
        'temperature': temperature,
        'bp': f'{blood_pressure_systolic}/{blood_pressure_diastolic}',
        'spo2': spo2,
        'rr': respiratory_rate,
        'rgb': avg_rgb_values.tolist()
    })

@app.route('/report')
def generate_report():
    data = {
        'bpm': bpm,
        'temperature': temperature,
        'bp': f'{blood_pressure_systolic}/{blood_pressure_diastolic}',
        'spo2': spo2,
        'rr': respiratory_rate
    }
    
    bart_report = generate_bart_report(data)
    gpt2_report = generate_gpt2_report(data)
    
    # Combine both reports into one large report
    combined_report = f"--- BART Report ---\n{bart_report}\n\n--- GPT-2 Report ---\n{gpt2_report}"
    
    return jsonify({'report': combined_report})

@app.route('/')
def index():
    return render_template('index.html')

def generate_bart_report(data):
    bpm = data['bpm']
    temperature = data['temperature']
    spo2 = data['spo2']
    bp = data['bp']
    rr = data['rr']

    # Prepare the input text for BART model
    input_text = f"Generate a detailed health report based on the following data:\nHeart Rate: {bpm} bpm\nBody Temperature: {temperature} °C\nBlood Pressure: {bp} mmHg\nSpO2: {spo2}%\nRespiratory Rate: {rr} breaths/min\n"

    # Encode and generate text using BART
    inputs = bart_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=500, num_beams=4, length_penalty=2.0, early_stopping=True)
    
    # Decode the generated summary
    report = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return report

def generate_gpt2_report(data):
    bpm = data['bpm']
    temperature = data['temperature']
    spo2 = data['spo2']
    bp = data['bp']
    rr = data['rr']

    # Prepare the input text for GPT-2 model
    input_text = f"Provide a detailed medical report based on these values: Heart Rate: {bpm} bpm, Body Temperature: {temperature} °C, Blood Pressure: {bp} mmHg, SpO2: {spo2}%, Respiratory Rate: {rr} breaths/min"

    # Encode input and generate text using GPT-2
    inputs = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    output = gpt2_model.generate(inputs, max_length=300, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    
    # Decode the generated summary
    report = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return report

if __name__ == "__main__":
    app.run(debug=True)
