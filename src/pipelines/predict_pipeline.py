import sys 
import pandas as pd
import os

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from utils import load_object


class Prediction:
    def __init__(self):
        pass

    def predict(self, features):
        
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_object(filepath = model_path) # this load object will just load the pickle file
            preprocessor = load_object(filepath = preprocessor_path)
            data_scale = preprocessor.transform(features)
            preds = model.predict(data_scale)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
