from src.CancerClassification.config.configuration import ConfigurationManager
from src.CancerClassification.components.data_ingestion import DataIngestion
from src.CancerClassification import logger

STAGE_NAME = 'Data Ingestion Stage'

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.extract_zip_file()

if __name__ == '__main__':
    try:
        logger.info(f">>>>stage {STAGE_NAME} has started")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>stage {STAGE_NAME} has completed")
    except Exception as e:
        logger.exception(e)
        raise e
        
        

