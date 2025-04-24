# Dockerfile

FROM python:3.12-slim

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install dependencies with longer timeout and retries
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --retries=5 -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for visualizations
RUN mkdir -p visualizations

# Set the entrypoint to allow flexible commands
ENTRYPOINT ["python"]

# Default command if none is provided
CMD ["main.py"]
