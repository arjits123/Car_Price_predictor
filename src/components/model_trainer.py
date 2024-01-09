import os 
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_config_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig

    def initiate_model_trainer(self, clean_path):
        try:
            df = pd.read_csv(clean_path)

            y = df['Price']
            X = df.drop(columns='Price')

            categorical_col = ['name','company','fuel_type']

            # making the column transformer
            ohe = OneHotEncoder()
            ohe.fit(X[categorical_col])
            column_transform = make_column_transformer((OneHotEncoder(categories = ohe.categories_),categorical_col),remainder='passthrough')


            logging.info('Made Column transformer')

            scores = []
            for i in range(1000):

                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1, random_state = i)

                model = LinearRegression()
    
                pipe = make_pipeline(column_transform, model)
                fitted_pipe = pipe.fit(X_train, y_train)
                y_pred = fitted_pipe.predict(X_test)
                scores.append(r2_score(y_test,y_pred))

            save_object(
                file_path = self.model_trainer_config.trained_model_config_path,
                obj = pipe
            )
            
            logging.info('Fitted the model')

            max_index = np.argmax(scores)
            return scores[max_index], max_index
        
        except Exception as e:
            raise CustomException(e,sys)    
