import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path # Ensure Path is imported
from MLProject.config.configuration import ConfigurationManager 
from MLProject.entity.config_entity import DataTransformationConfig 
from MLProject import logger

# Define the base directory where artifacts are expected to be downloaded inside the container
# This must match the ML_ARTIFACTS_DIR environment variable in the Dockerfile
ML_ARTIFACTS_BASE_DIR = os.environ.get("ML_ARTIFACTS_DIR", "artifacts/downloaded_model")


class PredictionPipeline:
    def __init__(self):
        # ConfigurationManager is still used to get feature lists, etc., from params.yaml and schema.yaml
        self.config_manager = ConfigurationManager()
        self.data_transformation_config = self.config_manager.get_data_transformation_config() 

        # --- Load preprocessor and model from the downloaded artifact paths ---
        # Construct paths using the base directory set by the download script
        preprocessor_path = Path(ML_ARTIFACTS_BASE_DIR) / "preprocessor" / "preprocessor.joblib" # MLflow saves artifacts in subdirectories
        model_path = Path(ML_ARTIFACTS_BASE_DIR) / "model" / "model.joblib" # MLflow saves models in a 'model' subdirectory

        self.preprocessor = joblib.load(preprocessor_path) 
        self.model = joblib.load(model_path) 
        logger.info(f"PredictionPipeline initialized: preprocessor loaded from {preprocessor_path}, model loaded from {model_path}.")

        # These lists are used for consistent column handling (reindexing) before CT
        numerical_cols_from_params = self.data_transformation_config.numerical_cols
        categorical_cols_from_params = self.data_transformation_config.categorical_cols
        columns_to_log_transform_from_params = self.data_transformation_config.columns_to_log_transform

        self.num_cols_to_log_for_ct = [col for col in numerical_cols_from_params if col in columns_to_log_transform_from_params]
        self.num_cols_no_log_for_ct = [col for col in numerical_cols_from_params if col not in columns_to_log_transform_from_params]
        self.cat_cols_for_ct = categorical_cols_from_params

        # Combine them in the order they are given to the ColumnTransformer during training
        self.all_expected_ct_columns_ordered = self.num_cols_to_log_for_ct + \
                                               self.num_cols_no_log_for_ct + \
                                               self.cat_cols_for_ct

        # Debugging: Log the columns the CT expects
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

            # Date Feature Engineering - This MUST be consistent with training!
            if 'Date' in data_to_transform.columns:
                data_to_transform['Date'] = pd.to_datetime(data_to_transform['Date'], errors='coerce')
                data_to_transform['Year'] = data_to_transform['Date'].dt.year
                data_to_transform['Month'] = data_to_transform['Date'].dt.month
                data_to_transform['Day'] = data_to_transform['Date'].dt.day
                data_to_transform['DayOfWeek'] = data_to_transform['Date'].dt.dayofweek
                data_to_transform['IsWeekend'] = data_to_transform['DayOfWeek'].isin([5, 6]).astype(int)
                logger.debug("Date features engineered for prediction input.")

            # Drop columns that were handled as non-features in training.
            columns_to_drop_from_X_pred = self.data_transformation_config.columns_to_drop_after_feature_eng.copy()
            if 'AQI_Bucket' in columns_to_drop_from_X_pred:
                columns_to_drop_from_X_pred.remove('AQI_Bucket') 
            
            for col in columns_to_drop_from_X_pred:
                if col in data_to_transform.columns:
                    data_to_transform = data_to_transform.drop(columns=[col])
                    logger.debug(f"Dropped column '{col}' from prediction input.")

            # Debugging: Check data_to_transform state before reindex
            logger.debug(f"PredictionPipeline: data_to_transform columns BEFORE reindex: {data_to_transform.columns.tolist()}")
            logger.debug(f"PredictionPipeline: data_to_transform dtypes BEFORE reindex:\n{data_to_transform.dtypes}")

            # Reindex the DataFrame to match the EXACT columns and ORDER expected by the ColumnTransformer
            data_for_ct = data_to_transform.reindex(columns=self.all_expected_ct_columns_ordered)
            
            # Debugging: Check data_for_ct state after reindex
            logger.debug(f"PredictionPipeline: data_for_ct columns AFTER reindex: {data_for_ct.columns.tolist()}")
            logger.debug(f"PredictionPipeline: data_for_ct dtypes AFTER reindex:\n{data_for_ct.dtypes}")
            logger.debug(f"PredictionPipeline: data_for_ct head AFTER reindex:\n{data_for_ct.head()}")


            transformed_data = self.preprocessor.transform(data_for_ct)
            logger.info("Prediction input data transformed using loaded preprocessor.")
            logger.debug(f"PredictionPipeline: Transformed data shape: {transformed_data.shape}")
            logger.debug(f"PredictionPipeline: Transformed data sample (first 5 values): {transformed_data[0, :5]}") 

            prediction = self.model.predict(transformed_data)
            logger.info("Prediction made successfully.")
            logger.debug(f"PredictionPipeline: Raw model prediction (before inverse transform): {prediction[0]}")

            if self.data_transformation_config.target_column in self.data_transformation_config.columns_to_log_transform:
                prediction = np.expm1(prediction)
                logger.info("Inverse log1p transformation applied to prediction.")
                logger.debug(f"PredictionPipeline: Final prediction (after inverse transform): {prediction[0]}")
            return prediction

        except Exception as e:
            logger.exception(f"Error during prediction: {e}")
            raise e
