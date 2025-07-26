# Declare build arguments at the very top.
ARG MLFLOW_RUN_ID_ARG # Renamed ARGs slightly for clarity
ARG MLFLOW_TRACKING_URI_ARG
ARG MLFLOW_TRACKING_USERNAME_ARG
ARG MLFLOW_TRACKING_PASSWORD_ARG

# Base image - using a more current slim Python image based on Debian Bookworm.
FROM python:3.9-slim-bookworm

# Install essential system dependencies:
RUN apt update && \
    apt install -y jq unzip build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy and install Python dependencies first.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code (src/MLProject) and core application files.
COPY src/MLProject /app/src/MLProject/
COPY app.py /app/
COPY config /app/config/
COPY params.yaml /app/
COPY schema.yaml /app/

# Copy the MLflow artifact download script and the entrypoint script.
COPY ./download_ml_artifacts.py /app/
COPY ./entrypoint.sh /app/

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Define the target directory where MLflow artifacts will be downloaded.
ENV ML_ARTIFACTS_DIR /app/artifacts/downloaded_model
RUN mkdir -p ${ML_ARTIFACTS_DIR}

# Set MLflow environment variables for subsequent RUN commands and for the container's runtime.
# This ensures they are consistently available.
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI_ARG}
ENV MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME_ARG}
ENV MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD_ARG}
ENV MLFLOW_RUN_ID=${MLFLOW_RUN_ID_ARG} # Reference the renamed ARG here

# Run the Python script to download the model and preprocessor artifacts from MLflow.
# It will now directly read the ENV variables set above.
RUN python /app/download_ml_artifacts.py

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["python", "app.py"]
