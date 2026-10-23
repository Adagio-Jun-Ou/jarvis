FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r jarvis && useradd -r -g jarvis -d /home/jarvis -s /sbin/nologin jarvis

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper ownership
RUN mkdir -p logs /home/jarvis/.jarvis && chown -R jarvis:jarvis logs /home/jarvis/.jarvis /app

# Switch to non-root user
USER jarvis

# Expose port for web interface
EXPOSE 8080

# Environment variables
ENV OLLAMA_API_KEY=""
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/jarvis

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "run.py", "--interface", "web", "--port", "8080"]