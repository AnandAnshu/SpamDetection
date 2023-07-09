import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from src.utils import load_objects


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            
            model=load_objects(file_path=model_path)
            preprocessor=load_objects(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            if preds[0] == 'spam':
                preds[0] = 'Spam'
            else:
                preds[0] = 'Not Spam'

            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
            self,
            message: str
        ):

        self.message=message

    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "message": [self.message]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        