from flask import Flask,jsonify,render_template,request
from flask_cors import CORS,cross_origin
from src.CancerClassification.utils.common import decodeImage
from src.CancerClassification.pipeline.prediction import PredictionPipeline
from src.CancerClassification.constants import PARAMS_FILE_PATH
from pathlib import Path
import os 

os.putenv('LANG','en_US_UTF-8')
os.putenv('LC_CALL','en_US_UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = 'inputImage.jpg'
        self.classifier = PredictionPipeline(self.filename)


@app.route('/',methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/train',methods=['GET','POST'])
@cross_origin()
def trainroute():
    os.system('dvc repro')
    # os.system('python main.py')
    return 'Training Done Successfully'


@app.route('/predict',methods=['POST'])
@cross_origin()
def prediction():
    image = request.json['image']
    decodeImage(image,clApp.filename)
    result = clApp.classifier.prediction()
    return jsonify(result)


if __name__=='__main__':
    clApp = ClientApp()
    app.run(host='0.0.0.0',port=8080,debug=True)