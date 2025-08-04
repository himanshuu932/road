# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set a non-interactive frontend for package installations
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install Python and other system dependencies with cleanup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    wget \
    sed \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# --- FIX for Dependency Error ---
# Upgrade pip and remove the strict version for contourpy that causes issues on Linux.
RUN python3 -m pip install --upgrade pip
RUN sed -i 's/contourpy==1.3.3/contourpy/g' requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Download the model weights during the build
# IMPORTANT: This now uses your specific model file link.
RUN mkdir -p runs/detect/yolov8s_all_countries_custom/weights
RUN wget -O runs/detect/yolov8s_all_countries_custom/weights/best.pt "https://drive.google.com/uc?export=download&id=1Knm8KsQP1o1QOub818BCYWE6J4tIPBbs"

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
# Use gunicorn for a production-ready server with eventlet workers for Socket.IO
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "app:app"]
