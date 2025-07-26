import os
import mlflow
import joblib
import logging
import sys
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_artifacts(run_id: str, local_dir: str):
    """
    Downloads the model and preprocessor artifacts for a given MLflow Run ID.
    The local_dir argument specifies the *base* directory where artifacts should be saved.
    """
    logger.info(f"Attempting to download artifacts for MLflow Run ID: {run_id} to {local_dir}")
    
    # MLflow client automatically picks up MLFLOW_TRACKING_URI, USERNAME, PASSWORD from ENV
    client = mlflow.tracking.MlflowClient()

    try:
        # Check if the run exists
        run = client.get_run(run_id)
        if not run:
            logger.error(f"MLflow Run ID '{run_id}' not found.")
            return False

        # Define target local paths within the container's expected structure
        # These paths must match where PredictionPipeline expects to load them.
        # MLflow logs 'model' and 'preprocessor' as subdirectories.
        model_target_path = Path(local_dir) / "model" / "model.joblib"
        preprocessor_target_path = Path(local_dir) / "preprocessor" / "preprocessor.joblib"

        os.makedirs(model_target_path.parent, exist_ok=True) # Ensure target dir for model
        os.makedirs(preprocessor_target_path.parent, exist_ok=True) # Ensure target dir for preprocessor

        # Download the model artifact
        # mlflow.artifacts.download_artifacts will download the 'model' folder.
        # We then need to find the actual model file inside it.
        downloaded_model_folder = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model", # The artifact_path in MLflow UI
            dst_path=str(model_target_path.parent.parent), # Download 'model' folder into the base downloaded_model dir
            tracking_uri=mlflow.get_tracking_uri()
        )
        
        # Verify the downloaded model path and move the actual model file
        # The model file is typically inside 'model/' folder downloaded by MLflow
        actual_model_file_in_download = Path(downloaded_model_folder) / "model.joblib" # Assuming it's named model.joblib inside
        if not actual_model_file_in_download.exists():
            # Fallback for older MLflow versions or different naming conventions
            for root, _, files in os.walk(downloaded_model_folder):
                for f in files:
                    if f.endswith((".pkl", ".joblib")):
                        actual_model_file_in_download = Path(root) / f
                        break
                if actual_model_file_in_download.exists():
                    break
        
        if actual_model_file_in_download.exists():
            os.rename(str(actual_model_file_in_download), str(model_target_path))
            logger.info(f"Model downloaded and saved to: {model_target_path}")
        else:
            logger.error("Could not find actual model file within the downloaded MLflow 'model' artifact.")
            return False
        
        # Clean up the temporary 'model' subdirectory if it was created
        if Path(downloaded_model_folder).exists() and Path(downloaded_model_folder).is_dir():
            os.rmdir(downloaded_model_folder) # Should be empty after moving model file

        # Download the preprocessor artifact
        downloaded_preprocessor_file = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="preprocessor/preprocessor.joblib", # Full artifact path
            dst_path=str(preprocessor_target_path.parent), # Download into the 'preprocessor' subdirectory
            tracking_uri=mlflow.get_tracking_uri()
        )
        
        # Ensure it's named correctly in the final location
        if Path(downloaded_preprocessor_file).name != preprocessor_target_path.name:
             os.rename(downloaded_preprocessor_file, preprocessor_target_path)
        logger.info(f"Preprocessor downloaded and saved to: {preprocessor_target_path}")

        return True

    except Exception as e:
        logger.error(f"Error downloading MLflow artifacts: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # The run_id and artifact_base_dir are passed as environment variables by entrypoint.sh
    run_id = os.environ.get("MLFLOW_RUN_ID") 
    artifact_base_dir = os.environ.get("ML_ARTIFACTS_DIR") 
    
    # MLflow tracking URI and credentials are also read from environment variables by MLflow client
    # No need to explicitly read them here, just ensure they are set in the environment.

    if not run_id:
        logger.error("MLFLOW_RUN_ID environment variable not set. Cannot download artifacts.")
        sys.exit(1) 
    if not artifact_base_dir:
        logger.error("ML_ARTIFACTS_DIR environment variable not set. Cannot download artifacts.")
        sys.exit(1)

    logger.info(f"Starting artifact download for Run ID: {run_id} to directory: {artifact_base_dir}")
    if download_artifacts(run_id, artifact_base_dir):
        logger.info("ML artifacts downloaded successfully.")
    else:
        logger.error("Failed to download ML artifacts. Exiting.")
        sys.exit(1)
