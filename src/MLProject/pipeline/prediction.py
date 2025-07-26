import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os 
from MLProject.config.configuration import ConfigurationManager 
from MLProject.entity.config_entity import DataTransformationConfig 
from MLProject import logger

# Define the new base directory for downloaded artifacts within the container.
# This MUST match the `ML_ARTIFACTS_DIR` environment variable value set in entrypoint.sh and Dockerfile,
# and where download_ml_artifacts.py will save the files.
DOWNLOADED_ARTIFACTS_DIR = "artifacts/downloaded_model" 

class PredictionPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.data_transformation_config = self.config_manager.get_data_transformation_config() 

        # --- CRITICAL CHANGE FOR MLFLOW ARTIFACT DOWNLOAD ---
        # Load the preprocessor and model from the directory where they are downloaded
        # by the download_ml_artifacts.py script at container startup.
        self.preprocessor = joblib.load(Path(os.path.join(DOWNLOADED_ARTIFACTS_DIR, "preprocessor.joblib"))) 
        self.model = joblib.load(Path(os.path.join(DOWNLOADED_ARTIFACTS_DIR, "model.joblib"))) 
        logger.info("PredictionPipeline initialized: preprocessor and model loaded from downloaded MLflow artifacts.")

        numerical_cols_from_params = self.data_transformation_config.numerical_cols
        categorical_cols_from_params = self.data_transformation_config.categorical_cols
        columns_to_log_transform_from_params = self.data_transformation_config.columns_to_log_transform

        self.num_cols_to_log_for_ct = [col for col in numerical_cols_from_params if col in columns_to_log_transform_from_params]
        self.num_cols_no_log_for_ct = [col for col in numerical_cols_from_params if col not in columns_to_log_transform_from_params]
        self.cat_cols_for_ct = categorical_cols_from_params

        self.all_expected_ct_columns_ordered = self.num_cols_to_log_for_ct + \
                                               self.num_cols_no_log_for_ct + \
                                               self.cat_cols_for_ct

        # NEW DEBUGGING: Log the columns the CT expects
        logger.debug(f"PredictionPipeline: CT expects numerical columns to log: {self.num_cols_to_log_for_ct}")
        logger.debug(f"PredictionPipeline: CT expects numerical columns NOT to log: {self.num_cols_no_log_for_ct}")
        logger.debug(f"PredictionPipeline: CT expects categorical columns: {self.cat_cols_for_ct}")
        logger.debug(f"PredictionPipeline: All expected CT columns in order: {self.all_expected_ct_columns_ordered}")


    def predict(self, raw_input_data: pd.DataFrame) -> np.ndarray:
        try:
            data_to_transform = raw_input_data.copy()
            logger.info(f"Received raw input data for prediction. Shape: {data_to_transform.shape}")

            logger.debug(f"PredictionPipeline: raw_input_data dtypes (from Flask form):\n{raw_input_data.dtypes}")
            logger.debug(f"PredictionPipeline: raw_input_data head (from Flask form):\n{raw_input_data.head()}")

            if 'Date' in data_to_transform.columns:
                data_to_transform['Date'] = pd.to_datetime(data_to_transform['Date'], errors='coerce')
                data_to_transform['Year'] = data_to_transform['Date'].dt.year
                data_to_transform['Month'] = data_to_transform['Date'].dt.month
                data_to_transform['Day'] = data_to_transform['Date'].dt.day
                data_to_transform['DayOfWeek'] = data_to_transform['Date'].dt.dayofweek
                data_to_transform['IsWeekend'] = data_to_transform['DayOfWeek'].isin([5, 6]).astype(int)
                logger.debug("Date features engineered for prediction input.")

            columns_to_drop_from_X_pred = self.data_transformation_config.columns_to_drop_after_feature_eng.copy()
            if 'AQI_Bucket' in columns_to_drop_from_X_pred:
                columns_to_drop_from_X_pred.remove('AQI_Bucket') 
            
            for col in columns_to_drop_from_X_pred:
                if col in data_to_transform.columns:
                    data_to_transform = data_to_transform.drop(columns=[col])
                    logger.debug(f"Dropped column '{col}' from prediction input.")

            logger.debug(f"PredictionPipeline: data_to_transform columns BEFORE reindex: {data_to_transform.columns.tolist()}")
            logger.debug(f"PredictionPipeline: data_to_transform dtypes BEFORE reindex:\n{data_to_transform.dtypes}")

            data_for_ct = data_to_transform.reindex(columns=self.all_expected_ct_columns_ordered)
            
            logger.debug(f"PredictionPipeline: data_for_ct columns AFTER reindex: {data_for_ct.columns.tolist()}")
            logger.debug(f"PredictionPipeline: data_for_ct dtypes AFTER reindex:\n{data_for_ct.dtypes}")
            logger.debug(f"PredictionPipeline: data_for_ct head AFTER reindex:\n{data_for_ct.head()}")


            transformed_data = self.preprocessor.transform(data_for_ct)
            logger.info("Prediction input data transformed using loaded preprocessor.")
            logger.debug(f"PredictionPipeline: Transformed data shape: {transformed_data.shape}")
            logger.debug(f"PredictionPipeline: Transformed data sample (first 5 values): {transformed_data[0, :5]}") # Log a sample of transformed data

            prediction = self.model.predict(transformed_data)
            logger.info("Prediction made successfully.")
            logger.debug(f"PredictionPipeline: Raw model prediction (before inverse transform): {prediction[0]}") # Log raw prediction

            if self.data_transformation_config.target_column in self.data_transformation_config.columns_to_log_transform:
                prediction = np.expm1(prediction)
                logger.info("Inverse log1p transformation applied to prediction.")
                logger.debug(f"PredictionPipeline: Final prediction (after inverse transform): {prediction[0]}") # Log final prediction
            return prediction

        except Exception as e:
            logger.exception(f"Error during prediction: {e}")
            raise e
