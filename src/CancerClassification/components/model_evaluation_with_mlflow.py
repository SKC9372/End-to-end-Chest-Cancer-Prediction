import mlflow.keras
import mlflow
import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
from src.CancerClassification.utils.common import save_json
from src.CancerClassification.entity.config_entity import EvaluationConfig



class Evaluation:
    def __init__(self,config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
        )

        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation='bilinear'
        )

        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        self.test_datagenerator = test_datagenerator.flow_from_directory(
            directory = Path(self.config.test_data),
            shuffle=False,
            **dataflow_kwargs
        )
    
    @staticmethod
    def load_model(path:Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_to_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.test_datagenerator)
        self.save_score()
    
    def save_score(self):
        scores = {'loss':self.score[0],'accuracy':self.score[1]}
        save_json(path=Path("scores.json"),data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_registry_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({'loss':self.score[0],'accuracy':self.score[1]}
                               )
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model,"model",registered_model_name='VGG16Model')
            else:
                mlflow.keras.log_model(self.model,"model")


    