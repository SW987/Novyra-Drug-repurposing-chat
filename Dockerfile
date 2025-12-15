FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data/chroma data/docs

# Expose port
EXPOSE 8501

# Health check (disabled for Railway)
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Make start script executable
RUN chmod +x /app/start.sh

# Start command
CMD ["/app/start.sh"]
