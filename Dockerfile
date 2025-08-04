# 1. Base image with CUDA 12.1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 2. Avoid interactive prompts and set workdir
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 3. Install Python 3.11, pip, and small utils; then symlink python3 → 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 \
      python3-pip \
      wget \
      sed && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Copy in your requirements.txt
COPY requirements.txt .

# 5. Upgrade pip & remove the strict contourpy specifier
RUN python3 -m pip install --upgrade pip
RUN sed -i '/^contourpy\s*==/s/==.*//' requirements.txt

# 6. Install Python deps, including torch+cu121 from PyTorch’s wheel index
RUN python3 -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu121/torch_stable.html \
      --extra-index-url https://pypi.org/simple \
      -r requirements.txt

# 7. Copy your application code
COPY . .

# 8. Download your custom YOLOv8 weights
RUN mkdir -p runs/detect/yolov8s_all_countries_custom/weights && \
    wget -O runs/detect/yolov8s_all_countries_custom/weights/best.pt \
      "https://drive.google.com/uc?export=download&id=1Knm8KsQP1o1QOub818BCYWE6J4tIPBbs"

# 9. Expose and run via Gunicorn + Eventlet
EXPOSE 5000
CMD ["gunicorn", \
     "--worker-class", "eventlet", \
     "-w", "1", \
     "--bind", "0.0.0.0:5000", \
     "app:app"]
