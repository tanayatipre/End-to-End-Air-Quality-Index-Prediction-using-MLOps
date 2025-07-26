# Use a Python base image (ensure it matches your requirements.txt Python version)
FROM python:3.9-slim-buster

# Install necessary system packages
# jq is needed for parsing JSON output from mlflow runs list in GHA to get run_id
# unzip is needed for extracting artifacts if MLflow downloads them as zip
# build-essential for some Python packages that might need compilation
RUN apt update -y && apt install -y awscli unzip jq build-essential

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python source code for the application and the download script
COPY src/MLProject /app/src/MLProject
COPY app.py /app/
COPY config /app/config/
COPY params.yaml /app/
COPY schema.yaml /app/
COPY download_ml_artifacts.py /app/ # Copy the artifact download script
COPY entrypoint.sh /app/ # Copy the entrypoint script
RUN chmod +x /app/entrypoint.sh # Make entrypoint script executable

# Define an environment variable for the downloaded model directory
# This path is where download_ml_artifacts.py will save files, and predictions.py will load from
ENV ML_ARTIFACTS_DIR /app/artifacts/downloaded_model

# Create the directory to store downloaded artifacts
RUN mkdir -p ${ML_ARTIFACTS_DIR}

# --- Build-time arguments for MLflow credentials and run ID ---
# These are passed from the GitHub Actions workflow during 'docker build'
ARG MLFLOW_TRACKING_USERNAME
ARG MLFLOW_TRACKING_PASSWORD
ARG MLFLOW_RUN_ID
ARG GITHUB_REPOSITORY # To construct MLFLOW_TRACKING_URI at build-time if needed

# --- Set MLflow environment variables for the download script to use ---
# These need to be available during the RUN python download_ml_artifacts.py step
ENV MLFLOW_TRACKING_URI=https://dagshub.com/${GITHUB_REPOSITORY}.mlflow
ENV MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME}
ENV MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}
ENV MLFLOW_RUN_ID=${MLFLOW_RUN_ID}

# Run the script to download the model and preprocessor during the image build
# This makes the image self-contained with the model artifacts
RUN python download_ml_artifacts.py

# Expose the port your Flask app runs on
EXPOSE 8080

# Set the entrypoint script to run when the container starts
ENTRYPOINT ["/app/entrypoint.sh"]

# Set the default command for the entrypoint script
CMD ["python", "app.py"]
