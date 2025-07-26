import os
import pandas as pd
import numpy as np # Ensure numpy is imported
import joblib
import mlflow
# Removed mlflow.sklearn as its direct log_model causing issues is replaced
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from MLProject import logger
from MLProject.utils.common import save_json # Ensure save_json is imported
from MLProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path # Ensure Path is imported

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path) # This will load the CatBoost model

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column] 

        # Apply inverse log1p transformation to actuals and predictions for evaluation if target was transformed
        # Ensure consistency with DataTransformation stage for AQI
        if self.config.target_column in self.config.all_params and self.config.target_column in self.config.columns_to_log_transform:
            # Assuming self.config.columns_to_log_transform is available via ConfigBox correctly
            # Check if target_column is in the list of columns that were log-transformed during data_transformation
            logger.info("Raw model predictions (log-transformed) obtained.")
            # For evaluation, inverse transform both test_y and model predictions back to original scale
            predicted_qualities_raw = model.predict(test_x)
            
            # Since eval_metrics expects original scale, apply inverse transform if target was logged
            test_y_original_scale = np.expm1(test_y)
            predicted_qualities_original_scale = np.expm1(predicted_qualities_raw)
            logger.info("Inverse log1p transformation applied to both actuals and predictions for evaluation.")

            # Use original scale for metrics calculation
            (rmse, mae, r2) = self.eval_metrics(test_y_original_scale, predicted_qualities_original_scale)
        else:
            # If target was not log-transformed, use raw predictions and actuals
            predicted_qualities = model.predict(test_x)
            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run(): # Note: this will create a nested run if called from main.py's run
            # You can also use mlflow.active_run() to get the existing run if main.py is already active
            
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)
            logger.info(f"Metrics saved locally to {self.config.metric_file_name}")

            mlflow.log_params(self.config.all_params) # Logs parameters from params.yaml
            logger.info("Model parameters logged to MLflow.")

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            logger.info("Metrics logged to MLflow.")

            # --- FIX FOR "unsupported endpoint" ERROR IN MODEL EVALUATION ---
            # Instead of mlflow.sklearn.log_model with registered_model_name,
            # log the model.joblib as a generic artifact.
            # model_path comes from config, so it's the path to the saved .joblib file.
            mlflow.log_artifact(local_path=str(self.config.model_path), artifact_path="evaluated_model") 
            logger.info("Model logged as MLflow artifact (under 'evaluated_model' path).")
            # The model is also logged by model_trainer. This one is for evaluation context.

            # Optional: Log the preprocessor here too if needed, but ModelTrainer already does this.
            # If you want to ensure the preprocessor is always with the evaluation, you can add it:
            # preprocessor_path = self.config.root_dir.parent.parent / "data_transformation" / "preprocessor.joblib"
            # if Path(preprocessor_path).exists():
            #     mlflow.log_artifact(local_path=str(preprocessor_path), artifact_path="evaluated_preprocessor")
            #     logger.info("Preprocessor logged as MLflow artifact (under 'evaluated_preprocessor' path).")

