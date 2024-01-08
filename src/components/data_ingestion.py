"""
Data ingestion is the process of importing, loading, or bringing data into a system or environment for processing. 
It involves retrieving data from various sources, such as databases, files, APIs, or external systems.
"""

import os 
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from dataclasses import dataclass

# importing for data transformation
from data_transformation import DataCleaning
from model_trainer import ModelTrainer

@dataclass #if this dataclass decorator is used we can directly define our class variable without using __init__

# Created this class to get any type of input required
class DataIngestionConfig:
    #These are the inputs we are giving to the data ingestion component
    raw_data_path: str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # the above three paths will be save in this variable, in short a instance is created

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            #Reading the data from the csv files
            df = pd.read_csv('data/quikr_car.csv')

            logging.info('Read the csv file in a dataframe')

            #Creating folder arifacts 
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)

            #create raw data path csv file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Ingestion of the data is completed') #logging

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    # Data ingestion
    obj = DataIngestion()
    raw_data = obj.initiate_data_ingestion()

    # Data Transformation
    data_clean = DataCleaning()
    clean_path = data_clean.data_cleaning(raw_data)

    #Model training 
    model_train = ModelTrainer()
    print(model_train.initiate_model_trainer(clean_path))








