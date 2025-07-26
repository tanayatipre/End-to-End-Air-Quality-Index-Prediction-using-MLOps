# Base image - using a more current slim Python image based on Debian Bookworm.
FROM python:3.9-slim-bookworm

# Install essential system dependencies:
# jq: for parsing JSON (e.g., MLflow run IDs)
# unzip: for extracting data archives
# build-essential: provides tools needed to compile some Python packages
# curl: often useful for downloading things.
RUN apt update && \
    apt install -y jq unzip build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies first.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and scripts.
# Only copy files needed for the application to run.
COPY src/MLProject /app/src/MLProject/
COPY app.py /app/
COPY config /app/config/
COPY params.yaml /app/
COPY schema.yaml /app/
COPY templates /app/templates/ # Ensure templates are copied
COPY static /app/static/     # Ensure static assets are copied

# Copy the MLflow artifact download script and the entrypoint script.
COPY ./download_ml_artifacts.py /app/
COPY ./entrypoint.sh /app/

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Define the target directory where MLflow artifacts will be downloaded at runtime.
ENV ML_ARTIFACTS_DIR /app/artifacts/downloaded_model
RUN mkdir -p ${ML_ARTIFACTS_DIR} # Create the directory inside the Docker image

# Set the container's entrypoint script.
# This script will be executed when the container starts.
ENTRYPOINT ["/app/entrypoint.sh"]

# Set the default command to execute (passed as arguments to ENTRYPOINT).
# This is typically your Flask app.
CMD ["python", "app.py"]
