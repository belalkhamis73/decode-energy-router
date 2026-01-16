# Stage 1: Builder (Compiles dependencies)
FROM python:3.10-slim as builder

WORKDIR /app

# Install system build dependencies (required for some SciML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker Layer Caching
COPY requirements.txt .

# Install Python deps into a virtual env to keep them isolated
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runner (The clean, production image)
FROM python:3.10-slim as runner

WORKDIR /app

# Create a non-root user for security (NERC CIP Best Practice)
RUN useradd -m energyuser
USER energyuser

# Copy installed dependencies from Builder stage
COPY --from=builder /root/.local /home/energyuser/.local

# Ensure scripts are in PATH
ENV PATH=/home/energyuser/.local/bin:$PATH

# Copy application code
COPY --chown=energyuser:energyuser . .

# Expose the API port
EXPOSE 8000

# Healthcheck: The grid needs to know if the Twin is alive
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to launch the Active Energy Router
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
