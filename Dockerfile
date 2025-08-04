# 1) Use a lightweight, official Python 3.10 base
FROM python:3.10-slim

# 2) Avoid interactive prompts, set working dir
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 3) Install apt deps (including libGL) and cleanup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget \
      sed \
      build-essential \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4) Copy requirements.txt
COPY requirements.txt .

# 5) Upgrade pip and strip out GPU/CUDA-specific pins
RUN python3 -m pip install --upgrade pip && \
    sed -i '/^torch==/d; /^torchvision==/d; /^torchaudio==/d; /^contourpy\s*==/s/==.*//; /^[[:space:]]*scipy==/d' requirements.txt

# 6) Install CPU-only PyTorch + Gunicorn + Eventlet
RUN python3 -m pip install --no-cache-dir \
      torch==2.5.1 \
      torchvision==0.20.1 \
      torchaudio==2.5.1 \
      gunicorn \
      eventlet \
      -r requirements.txt

# 7) Copy the rest of your app
COPY . .

# 8) Download your custom YOLOv8 weights
RUN mkdir -p runs/detect/train2/weights && \
    wget -O runs/detect/train2/weights/best.pt \
      "https://drive.google.com/uc?export=download&id=1Knm8KsQP1o1QOub818BCYWE6J4tIPBbs"

# 9) Expose and launch with Gunicorn + Eventlet
EXPOSE 5000
CMD ["gunicorn", \
     "--worker-class","eventlet", \
     "-w","1", \
     "--bind","0.0.0.0:5000", \
     "app:app"]