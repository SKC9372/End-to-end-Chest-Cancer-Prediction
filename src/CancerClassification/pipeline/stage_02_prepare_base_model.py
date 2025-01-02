from src.CancerClassification.config.configuration import ConfigurationManager
from src.CancerClassification.components.prepare_base_model import PrepareBaseModel
from src.CancerClassification import logger

STAGE_NAME = "PrepareBaseModel"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_prepare_base_model_config()
        self.prepare_base_model = PrepareBaseModel(config=self.config)

    def main(self):
        logger.info(f"Started stage: {STAGE_NAME}")
        self.prepare_base_model.get_base_model()
        self.prepare_base_model.update_base_model()
        logger.info(f"Completed stage: {STAGE_NAME}")


if __name__ == "__main__":
    try:
        logger.info(f'Starting {STAGE_NAME} stage')
        pipeline = PrepareBaseModelTrainingPipeline()
        pipeline.main()
        logger.info(f'Completed {STAGE_NAME} stage')
    except Exception as e:
        logger.exception(e)
        raise e
    
        
