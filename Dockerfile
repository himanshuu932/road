# Use a slim, official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install the essential system library, just in case
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies from your requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- THE FINAL FIX ---
# Force reinstall of the headless version to overwrite any conflicts
RUN pip install --no-deps --force-reinstall opencv-python-headless==4.12.0.88

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "app:app"]