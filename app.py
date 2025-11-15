import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
from twilio.rest import Client

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
YOLO_MODEL_PATH = 'yolov8s.pt'  # smaller/faster model
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(YOLO_MODEL_PATH)

# Twilio configuration - replace with your real credentials
ACCOUNT_SID = 'AC9112337a9e7e162f31b917561d352bbd'
AUTH_TOKEN = 'e77dca0da9bb8cd5bc18934e93e0879c'
TWILIO_PHONE = '+12314409475'
TARGET_PHONE = '+91 95155 89057'
twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

alert_sent = False  # To avoid spamming alerts

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def send_alert(message_body):
    global alert_sent
    if not alert_sent:
        try:
            message = twilio_client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE,
                to=TARGET_PHONE
            )
            print(f"Alert sent: {message.sid}")
            alert_sent = True
        except Exception as e:
            print(f"Failed to send alert: {e}")
def process_frame(frame):
    # Run YOLO with tuned confidence and using medium model
    results = model(frame, conf=0.05, iou=0.25, classes=[0])[0]
    boxes = [box for box in results.boxes if int(box.cls[0]) == 0]
    person_count = len(boxes)
    annotated_frame = results.plot()

    if person_count > 0:
        send_alert(f"Alert! Detected {person_count} persons - possible hostage/disaster situation.")
    return annotated_frame, person_count


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part in the request')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # After save, redirect to processing display page
        return redirect(url_for('process_file', filename=filename))
    else:
        flash('Allowed file types: png, jpg, jpeg, mp4, avi, mov')
        return redirect(request.url)

def generate_frames_image(path):
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (640, 480))
    annotated_frame, _ = process_frame(frame)
    ret, buffer = cv2.imencode('.jpg', annotated_frame)
    frame_bytes = buffer.tobytes()
    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_frames_video(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        annotated_frame, _ = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            break
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/process/<filename>')
def process_file(filename):
    file_ext = filename.rsplit('.', 1)[1].lower()
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if file_ext in {'mp4', 'avi', 'mov'}:
        # Stream video frames as MJPEG
        return Response(generate_frames_video(file_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif file_ext in {'png', 'jpg', 'jpeg'}:
        # Stream single annotated image as MJPEG for consistency
        return Response(generate_frames_image(file_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return f"Unsupported file type: {file_ext}", 400

if __name__ == '__main__':
    try:
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error starting server: {e}")
        raise

