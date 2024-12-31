from src.CancerClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.CancerClassification import logger

STAGE_NAME = 'Data Ingestion Stage'


try:
    logger.info(f">>>>stage {STAGE_NAME} has started")
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()
    logger.info(f">>>>stage {STAGE_NAME} has completed")
except Exception as e:
    logger.exception(e)