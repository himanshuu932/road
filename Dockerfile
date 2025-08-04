# Use NVIDIA base image or Ubuntu if on CPU
FROM python:3.10-slim

# Set YOLO config env (to avoid /root permission issue)
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install OpenCV dependencies + libGL
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Gunicorn with extended timeout for heavy detection
CMD ["gunicorn", "--worker-class", "eventlet", "--timeout", "120", "-w", "1", "--bind", "0.0.0.0:5000", "app:app"]
