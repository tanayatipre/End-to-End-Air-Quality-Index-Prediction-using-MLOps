import os
import mlflow
import joblib
import logging
import sys # Import sys for sys.exit

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
        model_target_path = os.path.join(local_dir, "model.joblib")
        preprocessor_target_path = os.path.join(local_dir, "preprocessor.joblib")

        os.makedirs(os.path.dirname(model_target_path), exist_ok=True) # Ensure target dir for model
        os.makedirs(os.path.dirname(preprocessor_target_path), exist_ok=True) # Ensure target dir for preprocessor

        # Download the model artifact
        # mlflow.sklearn.log_model logs the model into its own 'model' artifact folder.
        # We need to download the 'model' folder, then extract the actual model file.
        model_artifact_subdir = "model" 
        downloaded_model_folder_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=model_artifact_subdir,
            dst_path=os.path.dirname(model_target_path), # Download to parent dir of model.joblib
            tracking_uri=mlflow.get_tracking_uri() # Use the configured tracking URI
        )
        
        # After downloading 'model' directory, the actual model file might be named something like 'model.pkl'
        # or similar inside that 'model' directory. Find it and move it.
        model_file_found = False
        for root, _, files in os.walk(downloaded_model_folder_path):
            for f in files:
                if f.endswith((".pkl", ".joblib")): 
                    os.rename(os.path.join(root, f), model_target_path)
                    logger.info(f"Model downloaded and saved to: {model_target_path}")
                    model_file_found = True
                    break
            if model_file_found:
                break
        
        if not model_file_found:
            logger.error("Could not find actual model file within the downloaded MLflow 'model' artifact.")
            return False

        # Clean up the temporary 'model' subdirectory if it was created
        if os.path.exists(downloaded_model_folder_path) and os.path.isdir(downloaded_model_folder_path):
            os.rmdir(downloaded_model_folder_path) # Should be empty after moving model file

        # Download the preprocessor artifact
        # We logged it as 'preprocessor/preprocessor.joblib'
        preprocessor_artifact_path = "preprocessor/preprocessor.joblib" 
        downloaded_preprocessor_file = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=preprocessor_artifact_path,
            dst_path=os.path.dirname(preprocessor_target_path), # Download to parent dir of preprocessor.joblib
            tracking_uri=mlflow.get_tracking_uri()
        )
        
        # If mlflow.artifacts.download_artifacts downloads directly to the file,
        # we ensure it's in the final location with correct name.
        if os.path.basename(downloaded_preprocessor_file) != "preprocessor.joblib":
             os.rename(downloaded_preprocessor_file, preprocessor_target_path)
        logger.info(f"Preprocessor downloaded and saved to: {preprocessor_target_path}")

        return True

    except Exception as e:
        logger.error(f"Error downloading MLflow artifacts: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # The run_id should be passed as an environment variable
    # The artifact_base_dir specifies where PredictionPipeline expects artifacts.
    run_id = os.environ.get("MLFLOW_RUN_ID") 
    artifact_base_dir = os.environ.get("ML_ARTIFACTS_DIR", "artifacts/model_artifacts") # Default if not set
    
    if not run_id:
        logger.error("MLFLOW_RUN_ID environment variable not set. Cannot download artifacts.")
        sys.exit(1) # Use sys.exit for clean exit in scripts

    logger.info(f"Starting artifact download for Run ID: {run_id} to directory: {artifact_base_dir}")
    if download_artifacts(run_id, artifact_base_dir):
        logger.info("ML artifacts downloaded successfully.")
    else:
        logger.error("Failed to download ML artifacts. Exiting.")
        sys.exit(1)
