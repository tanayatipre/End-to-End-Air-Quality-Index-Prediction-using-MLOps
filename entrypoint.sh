#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Set MLflow tracking URI and credentials for the container's environment
# These are passed from GitHub Actions secrets/env variables
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" # From GitHub Actions env
export MLFLOW_TRACKING_USERNAME="${MLFLOW_TRACKING_USERNAME}" # From GitHub Actions env
export MLFLOW_TRACKING_PASSWORD="${MLFLOW_TRACKING_PASSWORD}" # From GitHub Actions env

# The MLFLOW_RUN_ID is passed as an environment variable to the container
# from the 'docker run' command in the GitHub Actions workflow.
# The ML_ARTIFACTS_DIR is also passed for the download script.

echo "Attempting to download ML artifacts for Run ID: ${MLFLOW_RUN_ID}"
echo "Artifacts will be saved to: ${ML_ARTIFACTS_DIR}"

# Run the Python script to download artifacts
python /app/download_ml_artifacts.py

echo "ML artifacts download finished. Starting Flask app."

# Start your Flask application
exec python /app/app.py
