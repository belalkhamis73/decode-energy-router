# Stage 1: Builder (Compiles dependencies)
FROM python:3.10-slim as builder

WORKDIR /tmp

# Install system build dependencies (required for SciML/PyTorch extensions)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker Layer Caching
COPY requirements.txt .

# Install dependencies into the user directory to allow easy copying
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runner (The clean, secure production image)
FROM python:3.10-slim as runner

WORKDIR /app

# Create a non-root user for security (NERC CIP Compliance)
RUN useradd -m energyuser
USER energyuser

# Copy installed Python packages from Builder stage
COPY --from=builder /root/.local /home/energyuser/.local

# Ensure scripts are in PATH
ENV PATH=/home/energyuser/.local/bin:$PATH

# Copy application code with correct ownership
COPY --chown=energyuser:energyuser . .

# Expose the API port
EXPOSE 8000

# Native Python Healthcheck (No curl required - keeps image minimal)
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Launch the Active Energy Router
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
