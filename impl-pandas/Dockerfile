# FHFA Seasonal Adjustment Pipeline Docker Image

FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY main.py .
COPY README.md .

# Create necessary directories
RUN mkdir -p data output .cache .checkpoints .metrics

# Set environment variables
ENV FHFA_ENV=production
ENV PYTHONPATH=/app

# Create non-root user
RUN useradd -m -s /bin/bash fhfa && \
    chown -R fhfa:fhfa /app
USER fhfa

# Default command
CMD ["python", "main.py", "--help"]