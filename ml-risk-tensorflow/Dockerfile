FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create empty __init__ files
RUN touch model/__init__.py inference/__init__.py

# Expose port
EXPOSE 5002

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5002

# Suppress TensorFlow warnings
ENV TF_CPP_MIN_LOG_LEVEL=2

# Run the server
CMD ["python", "-m", "inference.server"]
