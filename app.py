import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import torch
import eventlet

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

# --- Route ---
@app.route('/')
def index():
    return render_template('index.html')

# --- SocketIO Event for Frame Processing ---
@socketio.on('image')
def handle_image(data):
    if model is None:
        return

    try:
        # Decode base64 image
        image_data = data['image']
        threshold = float(data['threshold'])

        sbuf = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(sbuf, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run YOLO detection
        results = model(frame, verbose=False, conf=threshold)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0].cpu().numpy()]
                confidence = float(box.conf[0].cpu().numpy())
                cls_name = model.names[int(box.cls)]

                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'class': cls_name,
                    'confidence': confidence
                })

        if detections:
            emit('response', {'detections': detections})

    except Exception as e:
        print(f"[ERROR] During image processing: {e}")

# --- Run App ---
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
