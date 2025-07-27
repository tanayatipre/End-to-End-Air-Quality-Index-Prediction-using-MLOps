ARG MLFLOW_RUN_ID
ARG MLFLOW_TRACKING_URI
ARG MLFLOW_TRACKING_USERNAME
ARG MLFLOW_TRACKING_PASSWORD

FROM python:3.9-slim-bookworm

RUN apt update && \
    apt install -y jq unzip build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/MLProject /app/src/MLProject/
COPY app.py /app/
COPY config /app/config/
COPY params.yaml /app/
COPY schema.yaml /app/
COPY templates /app/templates/
COPY static /app/static/
COPY ./download_ml_artifacts.py /app/
COPY ./entrypoint.sh /app/

RUN chmod +x /app/entrypoint.sh

ENV ML_ARTIFACTS_DIR=/app/artifacts/downloaded_model
RUN mkdir -p ${ML_ARTIFACTS_DIR}

ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME}
# ENV MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}  # Avoid if secret
ENV MLFLOW_RUN_ID=${MLFLOW_RUN_ID}


ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "app.py"]
