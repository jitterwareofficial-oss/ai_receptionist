# Use an official, stable, and slim Python runtime as the base image
FROM python:3.11-slim

# Set system-level environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Set the working directory inside the container
WORKDIR /app

# Install bare-minimum system dependencies required for compiling certain Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency manifest first to leverage Docker's caching layers
COPY requirements.txt .

# Install Python packages securely, disabling cache to keep the image slim
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application source code into the container
COPY . .

# Create a non-privileged system user for runtime isolation (Security Guardrail)
RUN useradd -u 8888 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the network port your FastAPI server binds to
EXPOSE 5000

# Execute the application server using Uvicorn with optimized production workers
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2"]
