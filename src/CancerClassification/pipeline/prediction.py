import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from src.CancerClassification.utils.common import read_yaml
from src.CancerClassification.constants import PARAMS_FILE_PATH
from pathlib import Path
import os

params = read_yaml(PARAMS_FILE_PATH)

size = tuple(params.IMAGE_SIZE[:-1])
class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename

    def prediction(self):
        model = load_model(os.path.join('model','model.h5'))

        imagename = self.filename
        test_img = image.load_img(imagename,target_size = size )
        test_image = image.img_to_array(test_img)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        prediction = ''

        if result[0]==0:
            prediction = 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib'
        elif result[0]==1:
            prediction = 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa'
        elif result[0]==2:
            prediction = 'normal'
        else:
            prediction = 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'

        return [{'image':prediction}]