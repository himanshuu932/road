import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, send_file
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import torch
import tempfile

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_screenshare_key!'
socketio = SocketIO(app, async_mode='eventlet')

# --- Load YOLOv8 Model ---
MODEL_PATH = os.path.join('src', 'runs', 'detect', 'yolov8s_all_countries_custom2', 'weights', 'best.pt')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH).to(device)
        print("[INFO] YOLOv8 model loaded successfully.")
    else:
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        model = None
except Exception as e:
    print(f"[ERROR] Loading model failed: {e}")
    model = None

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if model is None:
        return "Model not loaded", 500

    file = request.files.get('file')
    threshold = float(request.form.get('threshold', 0.5))
    if not file:
        return "No file uploaded", 400

    filename = file.filename.lower()

    # --- Process image ---
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        processed_frame = process_frame(frame, threshold)

       
        _, buffer = cv2.imencode('.jpg', processed_frame)
        return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}

    # --- Process video ---
    elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Save temporarily to disk for OpenCV
        temp_file_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_file_path)

        cap = cv2.VideoCapture(temp_file_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame, threshold)
            frames.append(frame)
        cap.release()

        # For simplicity, return first frame
        _, buffer = cv2.imencode('.jpg', frames[0])
        return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}

    else:
        return "Unsupported file type", 400

# --- SocketIO Event for Live Stream ---
@socketio.on('image')
def handle_image(data):
    if model is None:
        return

    try:
        image_data = data['image']
        threshold = float(data.get('threshold', 0.5))

        sbuf = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(sbuf, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        processed_frame = process_frame(frame, threshold)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        b64_frame = base64.b64encode(buffer).decode('utf-8')

        emit('response', {'detections': get_detections(frame, threshold), 'image': f"data:image/jpeg;base64,{b64_frame}"})
    except Exception as e:
        print(f"[ERROR] During image processing: {e}")

# --- Helper Functions ---
def process_frame(frame, threshold=0.5):
    """Run YOLOv8 inference and draw bounding boxes"""
    results = model(frame, verbose=False, conf=threshold)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].cpu().numpy()]
            cls_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0].cpu().numpy())

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def get_detections(frame, threshold=0.5):
    """Return detection data without modifying frame"""
    results = model(frame, verbose=False, conf=threshold)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0].cpu().numpy()]
            cls_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0].cpu().numpy())
            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'class': cls_name,
                'confidence': confidence
            })
    return detections

# --- Run App ---
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
