"""
Data transformation involves converting and modifying raw data into a suitable format for analysis or further processing. 
It includes tasks such as cleaning, filtering, aggregating, and feature engineering.
"""
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
# Creat this class to get any type of input required to the data transformation component
class DataCleanConfig:
    clean_data_path: str = os.path.join('artifacts', 'clean_data.csv')
    
class DataCleaner:
    data_cleaner_config_path= os.path.join('artifacts', 'cleaner.pkl')
    
"""
Data quality issues

1. Year has many non-year values and also it is object data type not integer
2. Price need to be converted to integer from object and there is one row with "Ask For price"
3. km_driven has to be converted from object to integer
4. km_driven has nan values 
5. km_driven has "kms" string in it and commas also
6. fuel type has nan values
7. keep first three words of name field

"""
class DataCleaning:
    def __init__(self):
        self.data_clean_config = DataCleanConfig()
        self.clean_config = DataCleaner()

    def data_cleaning(self, raw_path):
        try:
            
            df = pd.read_csv(raw_path)
            
            #----- YEAR --------#
            # Check if each element of year is numeric
            df = df[df['year'].str.isnumeric()]
            # year has object datatype not integer datatype
            df['year'] = df['year'].astype('int')

            #------ PRICE -------#
            #Removing 'Ask For Price' from the Price column
            df = df[df['Price']!='Ask For Price']
            df['Price'] = df['Price'].str.replace(',','').astype('int')

            #------ KMS Driven ------#
            #removing commna from the numbers
            df['kms_driven'] = df['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
            df = df.dropna(subset='kms_driven') # drop nan values from kms driven

            df = df[df['kms_driven'].str.isnumeric()]
            df['kms_driven'] = df['kms_driven'].astype('int') # converting obj into integer

            #------ FUEL TYPE -----#
            df = df.dropna(subset='fuel_type') # dropping null values

            # ------ NAME ------#
            # Getting first 3 words of the brand name
            df['name'] = df['name'].str.split(' ').str.slice(0,3).str.join(' ')
            df.reset_index(drop=True)

            # whose priece is less than 6 lacs
            df = df[df['Price'] < 6e6].reset_index(drop=True)

            os.makedirs(os.path.dirname(self.data_clean_config.clean_data_path), exist_ok = True)
            df.to_csv(self.data_clean_config.clean_data_path, index=False, header=True)

            save_object(
                file_path = self.clean_config.data_cleaner_config_path,
                obj = df
            )

            logging.info('Cleaned data file is created') #logging

            return (
                self.data_clean_config.clean_data_path
                )

        except Exception as e:
            raise CustomException(e,sys)