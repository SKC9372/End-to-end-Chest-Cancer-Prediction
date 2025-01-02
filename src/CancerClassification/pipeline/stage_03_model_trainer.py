from src.CancerClassification.config.configuration import ConfigurationManager
from src.CancerClassification.components.training_model import Training
from src.CancerClassification import logger 

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
    
    def main(self):
        config = self.config_manager.get_training_config()
        training = Training(config=config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()



if __name__ ==  '__main__':
    try:
        logger.info(f'Starting {STAGE_NAME} stage')
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        logger.info(f'Completed {STAGE_NAME} stage')
    except Exception as e:
        logger.exception(e)
        raise e