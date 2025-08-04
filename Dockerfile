# 1. Base image with CUDA 12.1 runtime
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 2. Non-interactive installs & workdir
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 3. Install Python 3.10, pip, wget & sed; symlink python3 → python3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.10 \
      python3-pip \
      wget \
      sed && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Copy in your requirements
COPY requirements.txt .

# 5. Upgrade pip & cleanup contourpy spec
RUN python3 -m pip install --upgrade pip && \
    sed -i '/^contourpy\s*==/s/==.*//' requirements.txt && \
    sed -i '/^torch==/d; /^torchvision==/d; /^torchaudio==/d' requirements.txt

# 6. Install torch, torchvision, torchaudio for CUDA 12.1 (built for cp310)
RUN python3 -m pip install --no-cache-dir \
      torch==2.5.1+cu121 \
      torchvision==0.20.1+cu121 \
      torchaudio==2.5.1+cu121 \
      --extra-index-url https://download.pytorch.org/whl/cu121/

# 7. Install all other Python deps
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# 8. Copy your app code
COPY . .

# 9. Download your custom YOLOv8 weights
RUN mkdir -p runs/detect/yolov8s_all_countries_custom/weights && \
    wget -O runs/detect/yolov8s_all_countries_custom/weights/best.pt \
      "https://drive.google.com/uc?export=download&id=1Knm8KsQP1o1QOub818BCYWE6J4tIPBbs"

# 10. Expose and launch with Gunicorn + Eventlet
EXPOSE 5000
CMD ["gunicorn", \
     "--worker-class","eventlet", \
     "-w","1", \
     "--bind","0.0.0.0:5000", \
     "app:app"]
