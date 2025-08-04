# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Avoid interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install Python 3.11, pip and other utilities; ensure python3 points to 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 \
      python3-pip \
      wget \
      sed && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and make pip the one for python3
COPY requirements.txt .

# Upgrade pip to latest
RUN python3 -m pip install --upgrade pip

# Remove strict contourpy version to avoid Python version conflicts
RUN sed -i '/^contourpy\s*==/s/==.*//' requirements.txt

# (Optional) Verify Python and pip versions
# RUN python3 --version && python3 -m pip --version

# Install all Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download your custom YOLOv8 weights
RUN mkdir -p runs/detect/yolov8s_all_countries_custom/weights && \
    wget -O runs/detect/yolov8s_all_countries_custom/weights/best.pt \
      "https://drive.google.com/uc?export=download&id=1Knm8KsQP1o1QOub818BCYWE6J4tIPBbs"

# Expose the port your app uses
EXPOSE 5000

# Run the app via Gunicorn with eventlet workers (for Socket.IO)
CMD ["gunicorn", \
     "--worker-class", "eventlet", \
     "-w", "1", \
     "--bind", "0.0.0.0:5000", \
     "app:app"]
