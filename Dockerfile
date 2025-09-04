FROM python:3.11-slim

WORKDIR /app

# Add this line to install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose the port Railway will use
EXPOSE 5000

# Run the app with Gunicorn
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "app:app"]