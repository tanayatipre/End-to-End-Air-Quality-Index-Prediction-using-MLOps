import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn # Still needed for other mlflow.sklearn functionalities potentially
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV 
from MLProject import logger
from MLProject.entity.config_entity import ModelTrainerConfig
from pathlib import Path # Ensure Path is imported for correct path handling

class ModelTrainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path) 

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        # Define parameter distribution for RandomizedSearchCV
        param_dist = {
            'iterations': [500, 1000, 1500, 2000],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'bagging_temperature': [0.0, 0.5, 1.0, 1.5, 2.0],
            'border_count': [32, 64, 128, 254],
            'random_strength': [1, 10, 20, 50],
            'early_stopping_rounds': [50, 100, 200]
        }

        # Ensure random_state for reproducibility
        base_model = CatBoostRegressor(
            random_seed=42,
            verbose=0, # Suppress verbose output during training within CV
            task_type="CPU",
            allow_writing_files=False # Fix for CatBoostError about working dir
        )
        
        # Start MLflow run for logging
        with mlflow.start_run() as run:
            run_id = run.info.run_id # Get the current run ID
            logger.info(f"MLflow Run ID: {run_id}")

            if self.config.perform_tuning:
                logger.info("Starting RandomizedSearchCV for CatBoostRegressor...")
                random_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_dist,
                    n_iter=self.config.n_iter_search,
                    cv=self.config.cv_folds,
                    scoring=self.config.scoring_metric,
                    n_jobs=2, # Using n_jobs=2 as discussed for stability
                    verbose=1,
                    random_state=42,
                    refit=True 
                )

                random_search.fit(train_x, train_y) 

                best_model = random_search.best_estimator_
                best_params = random_search.best_params_
                best_score = random_search.best_score_

                logger.info(f"RandomizedSearchCV completed. Best parameters: {best_params}")
                logger.info(f"Best CV {self.config.scoring_metric} score: {best_score:.4f}")

                mlflow.log_params(best_params)
                mlflow.log_metric(f"best_cv_score_{self.config.scoring_metric}", best_score)
                logger.info(f"Logged best parameters and best CV score to MLflow.")

            else: # If tuning is disabled
                logger.info("Tuning is disabled. Training CatBoostRegressor with default parameters...")
                best_model = base_model.set_params(**self.config.params)
                best_model.fit(train_x, train_y)
                best_params = self.config.params

                mlflow.log_params(best_params)
                logger.info(f"Logged default CatBoostRegressor parameters to MLflow: {best_params}")

            # Save the model and preprocessor locally first (for joblib load in pipeline/local testing)
            model_save_path = Path(self.config.root_dir) / self.config.model_name # Use Path object for joining
            joblib.dump(best_model, model_save_path)
            logger.info(f"Trained model saved locally to {model_save_path}")

            # NEW: Load the preprocessor that was saved by DataTransformation stage
            # Ensure self.config.root_dir is a Path object, then use its methods
            preprocessor_path = self.config.root_dir.parent / "data_transformation" / "preprocessor.joblib"
            preprocessor_obj = joblib.load(preprocessor_path) # Load the preprocessor

            # --- FIX FOR "unsupported endpoint" ERROR ---
            # Instead of mlflow.sklearn.log_model, log the joblib file as a generic artifact.
            # This avoids interaction with the Model Registry endpoint which may be unsupported.
            mlflow.log_artifact(local_path=str(model_save_path), artifact_path="model") # Log the model.joblib
            logger.info("Trained model logged as MLflow artifact (under 'model' path).")

            # Log the preprocessor as a separate artifact
            mlflow.log_artifact(local_path=str(preprocessor_path), artifact_path="preprocessor") # Log the preprocessor.joblib
            logger.info("Preprocessor logged as MLflow artifact (under 'preprocessor' path).")

        logger.info("Model training stage completed successfully.")
