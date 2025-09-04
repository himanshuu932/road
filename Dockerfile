FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port for Cloud Run
EXPOSE 5000


# Run the app with Gunicorn
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "app:app"]
