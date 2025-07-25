import pandas as pd
import os
from MLProject import logger
from sklearn.linear_model import ElasticNet
import joblib
from MLProject.entity.config_entity import ModelTrainerConfig
from catboost import CatBoostRegressor

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

        logger.info(f"Training CatBoostRegressor with parameters: {self.config.params}")

        model = CatBoostRegressor(**self.config.params)
        model.fit(train_x, train_y)
        
        logger.info("CatBoostRegressor model training completed.")

        # Save the trained model
        model_save_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(model, model_save_path)

        logger.info(f"Trained CatBoostRegressor model saved to {model_save_path}")        