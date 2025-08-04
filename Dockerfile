    # Use an official NVIDIA CUDA runtime as a parent image
    FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

    # Set the working directory in the container
    WORKDIR /app

    # Install Python and other system dependencies
    RUN apt-get update && apt-get install -y \
        python3.11 \
        python3-pip \
        wget \
        && rm -rf /var/lib/apt/lists/*

    # Copy the requirements file into the container
    COPY requirements.txt .

    # Install Python dependencies
    RUN pip3 install --no-cache-dir -r requirements.txt

    # Copy the rest of the application code into the container
    COPY . .

    # Download the model weights during the build
    # IMPORTANT: Replace the URL with your direct download link
    RUN mkdir -p runs/detect/yolov8s_all_countries_custom/weights
    RUN wget -O runs/detect/yolov8s_all_countries_custom/weights/best.pt "https://drive.google.com/file/d/1Knm8KsQP1o1QOub818BCYWE6J4tIPBbs/view?usp=sharing"

    # Expose the port the app runs on
    EXPOSE 5000

    # Define the command to run the application
    # Use gunicorn for a production-ready server with eventlet workers for Socket.IO
    CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "app:app"]
    