import tensorflow as tf
from pathlib import Path  
from src.CancerClassification.entity.config_entity  import TrainingConfig  

class Training:
    def __init__(self,config:TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):
        datagenerator_kwargs = dict(rescale=1.0/255)

        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = "bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
            )
        self.validation_data_generator = valid_datagenerator.flow_from_directory(
            directory = self.config.validation_data,
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_augmentation:
            train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                shear_range=0.2,
                width_shift_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_generator = valid_datagenerator

        self.train_generator = train_generator.flow_from_directory(
            directory=self.config.training_data,
            shuffle=True,
            **dataflow_kwargs

        )

    @staticmethod
    def save_model(path:Path,model:tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.validation_data_generator.samples // self.validation_data_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs = self.config.params_epoch,
            steps_per_epoch = self.steps_per_epoch,
            validation_steps = self.validation_steps,
            validation_data = self.validation_data_generator
            )
        
        self.save_model(path = self.config.trained_model_path,
                        model=self.model
                        )

        