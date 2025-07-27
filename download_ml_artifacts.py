import os
import mlflow
from mlflow.artifacts import download_artifacts
from pathlib import Path
import logging
import sys # Import sys for sys.exit()
import shutil # Import shutil for shutil.rmtree()

# Configure logger
logger = logging.getLogger(__name__)
# Ensure logger is configured if running standalone, otherwise MLProject's __init__.py configures it
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(levelname)s: %(message)s')

def download_artifacts_from_mlflow():
    """
    Downloads model and preprocessor artifacts from a specified MLflow run ID
    to a local directory, relying on environment variables for configuration.
    """
    mlflow_run_id = os.environ.get("MLFLOW_RUN_ID")
    ml_artifacts_dir = os.environ.get("ML_ARTIFACTS_DIR")
    
    # Ensure environment variables are set
    if not mlflow_run_id:
        logger.error("MLFLOW_RUN_ID environment variable not set. Cannot download artifacts.")
        return False
    if not ml_artifacts_dir:
        logger.error("ML_ARTIFACTS_DIR environment variable not set. Cannot download artifacts.")
        return False

    # Clean and normalize the target directory path
    # Path().as_posix() converts to forward slashes for cross-platform compatibility
    # .strip() removes any leading/trailing whitespace that could cause WinError 3
    ml_artifacts_dir_clean = Path(ml_artifacts_dir).as_posix().strip()

    logger.info(f"Starting artifact download for Run ID: {mlflow_run_id} to directory: {ml_artifacts_dir_clean}")

    try:
        # Define the base temporary download location for MLflow artifacts
        # MLflow downloads artifacts into a structure like <dst_path>/<artifact_path>/<files>
        temp_mlflow_download_base_path = Path(ml_artifacts_dir_clean) / ".temp_mlflow_download"
        
        # Ensure the base temporary download directory exists
        os.makedirs(temp_mlflow_download_base_path, exist_ok=True)
        
        logger.info(f"Attempting to download artifacts for MLflow Run ID: {mlflow_run_id} to {temp_mlflow_download_base_path}")

        # --- Download Model Artifact ---
        # artifact_path="model" means download the entire 'model' folder
        downloaded_model_folder = download_artifacts(
            run_id=mlflow_run_id,
            artifact_path="model", # The artifact_path in MLflow UI for the model folder
            dst_path=str(temp_mlflow_download_base_path), # Download 'model' folder into the temp_mlflow_download dir
            tracking_uri=mlflow.get_tracking_uri()
        )
        logger.info(f"Model artifact downloaded to: {downloaded_model_folder}")

        # Define final target path for the model.joblib
        model_target_path = Path(ml_artifacts_dir_clean) / "model.joblib"
        
        # Move the actual model.joblib file from inside the downloaded 'model' folder
        actual_model_file_in_download = Path(downloaded_model_folder) / "model.joblib" 
        if not actual_model_file_in_download.exists():
            # Fallback for older MLflow versions or different naming conventions if model.joblib isn't directly in 'model/'
            for root, _, files in os.walk(downloaded_model_folder):
                for f in files:
                    if f.endswith((".pkl", ".joblib")): # Check for .pkl or .joblib extensions
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

        # --- Download Preprocessor Artifact ---
        # artifact_path="preprocessor" means download the entire 'preprocessor' folder
        downloaded_preprocessor_folder = download_artifacts(
            run_id=mlflow_run_id,
            artifact_path="preprocessor", # The artifact_path in MLflow UI for the preprocessor folder
            dst_path=str(temp_mlflow_download_base_path), # Download 'preprocessor' folder into the temp_mlflow_download dir
            tracking_uri=mlflow.get_tracking_uri()
        )
        logger.info(f"Preprocessor artifact downloaded to: {downloaded_preprocessor_folder}")

        # Define final target path for the preprocessor.joblib
        preprocessor_target_path = Path(ml_artifacts_dir_clean) / "preprocessor.joblib"

        # Move the actual preprocessor.joblib file from inside the downloaded 'preprocessor' folder
        actual_preprocessor_file_in_download = Path(downloaded_preprocessor_folder) / "preprocessor.joblib" 
        if not actual_preprocessor_file_in_download.exists():
            # Fallback if preprocessor.joblib isn't directly in 'preprocessor/'
            for root, _, files in os.walk(downloaded_preprocessor_folder):
                for f in files:
                    if f.endswith((".pkl", ".joblib")):
                        actual_preprocessor_file_in_download = Path(root) / f
                        break
                if actual_preprocessor_file_in_download.exists():
                    break
        
        if actual_preprocessor_file_in_download.exists():
            os.rename(str(actual_preprocessor_file_in_download), str(preprocessor_target_path))
            logger.info(f"Preprocessor downloaded and saved to: {preprocessor_target_path}")
        else:
            logger.error(f"Preprocessor artifact not found at expected path: {preprocessor_src_path}") # Corrected var name
            return False

        # --- Clean up temporary download directory ---
        # This will remove the .temp_mlflow_download folder and all its contents
        if temp_mlflow_download_base_path.exists() and temp_mlflow_download_base_path.is_dir():
            shutil.rmtree(temp_mlflow_download_base_path) 
            logger.info(f"Cleaned up temporary download directory: {temp_mlflow_download_base_path}")

        return True

    except Exception as e:
        logger.error(f"Error downloading MLflow artifacts: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # The run_id and artifact_base_dir are passed as environment variables by entrypoint.sh
    run_id = os.environ.get("MLFLOW_RUN_ID") 
    artifact_base_dir = os.environ.get("ML_ARTIFACTS_DIR", "").strip()
    
    # MLflow tracking URI and credentials are also read from environment variables by MLflow client
    # No need to explicitly read them here, just ensure they are set in the environment.

    if not run_id:
        logger.error("MLFLOW_RUN_ID environment variable not set. Cannot download artifacts.")
        sys.exit(1) 
    if not artifact_base_dir:
        logger.error("ML_ARTIFACTS_DIR environment variable not set. Cannot download artifacts.")
        sys.exit(1)

    # Call the main download function
    success = download_artifacts_from_mlflow()
    if not success:
        logger.error("Failed to download ML artifacts. Exiting.")
        sys.exit(1)
