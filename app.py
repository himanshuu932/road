import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import torch
import eventlet

eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_screenshare_key!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

MODEL_PATH = os.path.join('runs', 'detect', 'train2', 'weights', 'best.pt')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = YOLO(MODEL_PATH).to(device)
        print("[INFO] YOLO model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
else:
    print(f"[ERROR] Model file not found at: {MODEL_PATH}")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(data):
    if not model:
        return

    image_data = data['image']
    threshold = float(data.get('threshold', 0.4))
    print(f"[INFO] Received frame with threshold: {threshold}")

    sbuf = base64.b64decode(image_data.split(',')[1])
    frame = cv2.imdecode(np.frombuffer(sbuf, np.uint8), cv2.IMREAD_COLOR)

    results = model(frame, conf=threshold, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0].cpu().numpy()]
            conf = float(box.conf[0].cpu().numpy())
            cls_name = model.names[int(box.cls)]
            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'class': cls_name,
                'confidence': conf
            })

    emit('response', {'detections': detections})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
