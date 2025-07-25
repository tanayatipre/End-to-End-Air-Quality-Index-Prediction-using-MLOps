from MLProject.config.configuration import ConfigurationManager
from MLProject.components.model_trainer import ModelTrainer
from MLProject import logger
from pathlib import Path

STAGE_NAME = "Model Trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer_obj = ModelTrainer(config=model_trainer_config)
            model_trainer_obj.train()
        except Exception as e:
            logger.exception(f"Error in Model Trainer Stage: {e}")
            raise e


if __name__=='__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e