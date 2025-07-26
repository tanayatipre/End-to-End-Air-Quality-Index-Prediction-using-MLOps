# Start with a Python base image. Choose a version that matches your requirements.txt
FROM python:3.9-slim-buster 

# Set the working directory inside the container
WORKDIR /app

# Install OS-level dependencies required for awscli and Python packages
# build-essential is for packages like numpy, pandas that might need compilation tools
# git might be needed by some pip packages or mlflow/dagshub internals
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    awscli \
    unzip \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/* # Clean up apt cache

# Copy requirements.txt and install Python dependencies.
# mlflow[s3] pulls boto3. dagshub also needs to be installed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    mlflow[s3] \ # Install mlflow with S3 support
    dagshub # Install dagshub

# Copy your application code and the artifact download script, and the entrypoint script
COPY src/MLProject /app/src/MLProject
COPY app.py /app/
COPY config /app/config/
COPY params.yaml /app/
COPY schema.yaml /app/
COPY templates /app/templates/
COPY static /app/static/
COPY download_ml_artifacts.py /app/ # Copy the new script
COPY entrypoint.sh /app/ # Copy the entrypoint script

# Create directories where the downloaded artifacts will be stored.
# This path must match the `local_dir` in download_ml_artifacts.py and
# where PredictionPipeline expects to load the models.
# This is a base directory; actual model/preprocessor will be in subdirs.
# Let's align this with PredictionPipeline's expected paths.
# If PredictionPipeline expects 'artifacts/model_trainer/model.joblib' and 'artifacts/data_transformation/preprocessor.joblib',
# then download_ml_artifacts.py should save directly to these paths.
# To keep download_ml_artifacts.py simpler with its 'local_dir' parameter:
# We'll create a single base directory for downloaded artifacts.
RUN mkdir -p /app/artifacts/downloaded_model 

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint to the shell script that downloads artifacts and then starts Flask
ENTRYPOINT ["/app/entrypoint.sh"]

# Expose the port your Flask app listens on
EXPOSE 8080

# CMD is typically the default arguments to the ENTRYPOINT (ignored if ENTRYPOINT is set)
# CMD ["python", "app.py"] # This will be called by entrypoint.sh
