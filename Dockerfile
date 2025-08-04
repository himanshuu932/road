FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.10 python3-pip wget sed && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip && \
    sed -i '/^contourpy\s*==/s/==.*//' requirements.txt && \
    sed -i '/^torch==/d; /^torchvision==/d; /^torchaudio==/d' requirements.txt && \
    sed -i '/^scipy\s*==/d' requirements.txt

RUN python3 -m pip install --no-cache-dir \
      torch==2.5.1+cu121 \
      torchvision==0.20.1+cu121 \
      torchaudio==2.5.1+cu121 \
      --extra-index-url https://download.pytorch.org/whl/cu121/

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p runs/detect/yolov8s_all_countries_custom/weights && \
    wget -O runs/detect/yolov8s_all_countries_custom/weights/best.pt \
      "https://drive.google.com/uc?export=download&id=1Knm8KsQP1o1QOub818BCYWE6J4tIPBbs"

EXPOSE 5000
CMD ["gunicorn","--worker-class","eventlet","-w","1","--bind","0.0.0.0:5000","app:app"]
