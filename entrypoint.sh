#!/bin/bash
set -e
set -x # Keep set -x for verbose execution logging

# Log the start of the entrypoint script
echo "--- Starting entrypoint.sh script ---"

# Export environment variables for MLflow artifact download script
# These are passed from docker run -e flags
echo "Setting MLflow environment variables for artifact download..."
export MLFLOW_TRACKING_URI
export MLFLOW_TRACKING_USERNAME
export MLFLOW_TRACKING_PASSWORD
export MLFLOW_RUN_ID
export ML_ARTIFACTS_DIR

echo "MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}"
echo "MLFLOW_RUN_ID: ${MLFLOW_RUN_ID}"
echo "ML_ARTIFACTS_DIR: ${ML_ARTIFACTS_DIR}"

# Attempt to download MLflow artifacts
echo "Attempting to download MLflow artifacts..."
python /app/download_ml_artifacts.py

# Check the exit code of the download script
if [ $? -ne 0 ]; then
    echo "ERROR: MLflow artifact download failed!"
    exit 1
else
    echo "MLflow artifacts downloaded successfully."
fi

# Confirm downloaded files exist before starting the app
echo "Verifying downloaded model and preprocessor files..."
if [ ! -f "${ML_ARTIFACTS_DIR}/model.joblib" ]; then
    echo "ERROR: model.joblib not found after download!"
    exit 1
fi
if [ ! -f "${ML_ARTIFACTS_DIR}/preprocessor.joblib" ]; then
    echo "ERROR: preprocessor.joblib not found after download!"
    exit 1
fi
echo "Model and preprocessor files confirmed."

echo "--- Starting the Flask application ---"
exec "$@"
