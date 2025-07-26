from MLProject.constants import *
from MLProject.utils.common import read_yaml, create_directories
from MLProject.entity.config_entity import (DataIngestionConfig,
                                            DataValidationConfig,
                                            DataTransformationConfig,
                                            ModelTrainerConfig,
                                            ModelEvaluationConfig)
from MLProject import logger

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        logger.info(f"Debug: Schema loaded into Configuration Manager:{self.schema.COLUMNS}")
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            csv_file_path=config.csv_file_path, 
            all_schema=schema
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.data_transformation
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            preprocessor_name=config.preprocessor_name,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            target_column=schema.name,
            numerical_cols=params.numerical_cols,
            categorical_cols=params.categorical_cols,
            columns_to_log_transform=params.columns_to_log_transform,
            columns_to_drop_after_feature_eng=params.columns_to_drop_after_feature_eng,
            test_size=params.test_size
        )

        return data_transformation_config
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        model_params = self.params.model_trainer.CatBoostRegressor # Default/initial model parameters
        tuning_params = self.params.model_trainer.tuning # NEW: Access tuning parameters
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir = config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            params = model_params, # Pass default parameters
            target_column = schema.name,
            # NEW: Pass tuning parameters
            perform_tuning = tuning_params.perform_tuning,
            n_iter_search = tuning_params.n_iter_search,
            cv_folds = tuning_params.cv_folds,
            scoring_metric = tuning_params.scoring_metric
        )

        return model_trainer_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        # Pass the same default/initial parameters to all_params for consistency with MLflow logging.
        # The ModelTrainer will log the *actual* best parameters if tuning occurs.
        params = self.params.model_trainer.CatBoostRegressor 
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params=params, 
            metric_file_name=config.metric_file_name,
            target_column=schema.name,
            mlflow_uri="https://dagshub.com/tanayatipre8/End-to-End-Machine-Learning-Project-with-MLFlow.mlflow"
        )

        return model_evaluation_config