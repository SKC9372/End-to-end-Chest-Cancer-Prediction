from src.CancerClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.CancerClassification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.CancerClassification.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from src.CancerClassification import logger

STAGE_NAME = 'Data Ingestion Stage'


try:
    logger.info(f">>>>stage {STAGE_NAME} has started")
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()
    logger.info(f">>>>stage {STAGE_NAME} has completed")
except Exception as e:
    logger.exception(e)

STAGE_NAME = "PrepareBaseModel"

try:
    logger.info(f"Started stage: {STAGE_NAME}")
    config = PrepareBaseModelTrainingPipeline()
    config.main()
    logger.info(f"Completed stage: {STAGE_NAME}")
except Exception as e:
    raise e

STAGE_NAME = "Training"

try:
    logger.info(f"Started stage: {STAGE_NAME}")
    pipeline = ModelTrainingPipeline()
    pipeline.main()
    logger.info(f"Completed stage: {STAGE_NAME}")
except Exception as e:
    raise e

