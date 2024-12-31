import os 
from box.exceptions import BoxValueError
import yaml
import json
import joblib
from src.CancerClassification import logger
from pathlib import Path
from typing import Any
from box import ConfigBox
import base64
from ensure import ensure_annotations

@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:
    """
    Reads a YAML file and returns a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as file:
          content = yaml.safe_load(file)
          logger.info(f" {path_to_yaml}:YAML file loaded successfully.")
          return ConfigBox(content)
    except BoxValueError:
       raise ValueError('Yaml file not found or is empty')
    except Exception as e:
       raise e
    
@ensure_annotations
def create_directories(path_to_directories:list,verbose=True):
   for path in path_to_directories:
      os.makedirs(path,exist_ok=True)
      if verbose:
         logger.info(f"Created directory: {path}")

@ensure_annotations
def save_json(path:Path,data:dict):
   with open(path, 'w') as f:
      json.dump(data, f, indent=4)
   logger.info(f"Data saved to JSON file: {path}")

@ensure_annotations
def load_json(path:Path) -> ConfigBox:
   with open(path) as f:
      data = json.load(f)
   logger.info(f"Data loaded from JSON file: {path}")
   return ConfigBox(data)


@ensure_annotations
def save_binary(data:Any,path:Path):
   joblib.dump(value=data,filename=path)
   logger.info(f"Data saved to binary file: {path}")


@ensure_annotations
def load_binary(path:Path) -> Any:
    data = joblib.load(path)
    logger.info(f"Data loaded from binary file: {path}")
    return data

@ensure_annotations
def get_size(path:Path) -> str:
   size = round(os.path.getsize(path)/1024)
   return f"{size} KB"

def decodeImage(imgstring,filename):
    imgdata = base64.b64decode(imgstring)
    with open(filename, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImage(image_path):
    with open(image_path, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    return my_string.decode('utf-8') 

   
