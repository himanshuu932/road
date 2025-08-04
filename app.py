import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import torch
import eventlet

# --- App and SocketIO Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_screenshare_key!'
socketio = SocketIO(app, async_mode='eventlet')

# --- Load Your Trained YOLOv8 Model ---
MODEL_PATH = os.path.join('runs', 'detect', 'train2', 'weights', 'best.pt')

# Check for GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print("YOLOv8 model loaded successfully.")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    """Renders the main page with the screen share feed."""
    return render_template('index.html')

@socketio.on('image')
@socketio.on('image')
def handle_image(data):
    """
    Receives an image frame and confidence threshold from the client,
    processes it, and sends back detection results.
    """
    if model is None:
        print("⚠️ Model not loaded. Skipping frame.")
        return

    # Decode base64 image
    image_data = data['image']
    sbuf = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(sbuf, dtype=np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Confidence threshold
    conf_threshold = float(data['threshold'])
    print(f"[INFO] Received frame. Threshold: {conf_threshold}")

    # --- Run YOLOv8 Inference ---
    results = model(frame, verbose=False, conf=conf_threshold, device=device)

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

    print(f"[INFO] Detections found: {len(detections)}")
    emit('response', {'detections': detections})

if __name__ == '__main__':
    # Run with eventlet
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
