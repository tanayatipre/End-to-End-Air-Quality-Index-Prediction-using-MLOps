from MLProject.config.configuration import ConfigurationManager
from MLProject.components.model_evaluation import ModelEvaluation
from MLProject import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try: 
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation_obj = ModelEvaluation(config=model_evaluation_config) # CORRECTED
            model_evaluation_obj.log_into_mlflow() # CORRECTED
        except Exception as e:
            logger.exception(f"Error in Model Evaluation Stage: {e}")
            raise e


if __name__=='__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e