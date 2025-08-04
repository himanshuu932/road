FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    wget \
    sed && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN sed -i '/^contourpy/s/==.*//' requirements.txt
RUN python3 --version && pip3 --version
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p runs/detect/yolov8s_all_countries_custom/weights && \
    wget -O runs/detect/yolov8s_all_countries_custom/weights/best.pt \
    "https://drive.google.com/uc?export=download&id=1Knm8KsQP1o1QOub818BCYWE6J4tIPBbs"

EXPOSE 5000
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "app:app"]
