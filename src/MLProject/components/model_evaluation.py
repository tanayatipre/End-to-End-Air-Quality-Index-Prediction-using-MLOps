import os
import pandas as pd
from MLProject import logger
# from sklearn.linear_model import ElasticNet # REMOVED: Unused import
import joblib
import mlflow.sklearn
import mlflow
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from MLProject.utils.common import save_json
from MLProject.entity.config_entity import ModelEvaluationConfig
from MLProject.utils.common import save_json # REMOVED: Duplicate import
from pathlib import Path


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
        test_y_log_transformed = test_data[self.config.target_column] 

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predicted_qualities_log_transformed = model.predict(test_x)
            logger.info("Raw model predictions (log-transformed) obtained.")

            test_y_original_scale = np.expm1(test_y_log_transformed)
            predicted_qualities_original_scale = np.expm1(predicted_qualities_log_transformed)
            logger.info("Inverse log1p transformation applied to both actuals and predictions for evaluation.")
            
            # Calculate metrics on the original scale
            (rmse, mae, r2) = self.eval_metrics(test_y_original_scale, predicted_qualities_original_scale)

            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)
            logger.info(f"Metrics saved locally to {self.config.metric_file_name}")

            mlflow.log_params(self.config.all_params) # Logs CatBoost parameters
            logger.info("Model parameters logged to MLflow.")

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            logger.info("Metrics logged to MLflow.")

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.sklearn.log_model(model, "model", registered_model_name="CatBoostModel") # CHANGED: Specific model name
                logger.info("Model registered in MLflow registry as 'CatBoostModel'.")
            else:
                mlflow.sklearn.log_model(model, "model")
                logger.info("Model logged to MLflow.")