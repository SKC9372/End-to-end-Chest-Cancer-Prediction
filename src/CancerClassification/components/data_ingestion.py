import os
from src.CancerClassification import  logger
import zipfile
from src.CancerClassification.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config

    def download_data(self):
        try:

            data_url = self.config.source_url

            zip_download_dir = self.config.local_data_file

            os.makedirs('artifacts/data_ingestion',exist_ok=True)

            logger.info(f"Downloading data from {data_url} to {zip_download_dir}")

            os.system(f'kaggle datasets download {data_url} -p {self.config.root_dir}')

            logger.info(f"Data downloaded at {zip_download_dir}")
        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        unzip_file_path = self.config.unzip_dir
        os.makedirs(unzip_file_path,exist_ok=True)

        zip_path = self.config.root_dir + '/' + self.config.source_url.split('/')[-1]

        with zipfile.ZipFile(zip_path+'.zip','r') as zip_file:
            zip_file.extractall(unzip_file_path)
            logger.info(f"Data extracted to {unzip_file_path}")
