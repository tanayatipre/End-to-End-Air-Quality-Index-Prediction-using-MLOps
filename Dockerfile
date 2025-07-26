# Declare build arguments at the very top.
# These will be passed from your GitHub Actions workflow.
ARG MLFLOW_RUN_ID
ARG MLFLOW_TRACKING_URI_ARG
ARG MLFLOW_TRACKING_USERNAME_ARG
ARG MLFLOW_TRACKING_PASSWORD_ARG

# Base image - using a more current slim Python image based on Debian Bookworm.
# This resolves the apt update 404 error by using active repositories.
FROM python:3.9-slim-bookworm

# Install essential system dependencies:
# - jq: for parsing JSON (e.g., MLflow run IDs)
# - unzip: for extracting data archives
# - build-essential: provides tools needed to compile some Python packages (e.g., numpy, scipy, lxml)
# - curl: often useful for downloading things.
# - rm -rf /var/lib/apt/lists/*: Cleans up apt cache to reduce image size.
RUN apt update && \
    apt install -y jq unzip build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy and install Python dependencies first.
# This optimizes Docker layering: if requirements.txt doesn't change, this layer is cached.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code (src/MLProject) and core application files (app.py, config, params, schema).
# Using './' explicitly for source ensures context is relative to the build root.
COPY src/MLProject /app/src/MLProject/
COPY app.py /app/
COPY config /app/config/
COPY params.yaml /app/
COPY schema.yaml /app/

# Copy the MLflow artifact download script and the entrypoint script.
# These files are expected to be at the root of your repository.
COPY ./download_ml_artifacts.py /app/
COPY ./entrypoint.sh /app/

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Define the target directory where MLflow artifacts (model, preprocessor) will be downloaded.
# This uses an environment variable to ensure consistency across scripts.
ENV ML_ARTIFACTS_DIR /app/artifacts/downloaded_model
RUN mkdir -p ${ML_ARTIFACTS_DIR} # Create the directory inside the Docker image

# Set MLflow environment variables at build time using the ARG values.
# These variables are consumed by 'download_ml_artifacts.py'.
# The correct syntax is ${ARG_NAME} for referencing ARGs in ENV instructions.
# NOTE: While necessary for build-time download, be aware that passing secrets (like DAGSHUB_TOKEN)
# via build ARGs and then ENV vars means they are present in the build layers.
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI_ARG}
ENV MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME_ARG}
ENV MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD_ARG}
ENV MLFLOW_RUN_ID=${MLFLOW_RUN_ID} # Removed inline comment to fix syntax error

# Run the Python script to download the model and preprocessor artifacts from MLflow.
# This happens during the Docker build process.
RUN python /app/download_ml_artifacts.py

# Set the container's entrypoint script.
# This script will be executed when the container starts and will then run CMD.
ENTRYPOINT ["/app/entrypoint.sh"]

# Set the default command to execute when the container starts.
# This command is passed to the ENTRYPOINT script.
CMD ["python", "app.py"]
