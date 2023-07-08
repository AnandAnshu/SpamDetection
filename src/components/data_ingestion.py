import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        #to read data from source(ex: database)
        logging.info("Entered the data ingestion methor or component")
        try:
            data=pd.read_csv('notebook\data\spam.tsv', sep='\t')
            logging.info("Fetched the dataset as dataframe")
            data.rename(columns={'label':'Target','message':'Text'},inplace=True)
            target_count = data.Target.value_counts()
            count_class_0, count_class_1 = data.Target.value_counts()
            target_class_0 = data[data['Target'] == "ham"]
            target_class_1 = data[data['Target'] == "spam"]
            data_class_0_under = target_class_0.sample(count_class_1)
            data_test_under = pd.concat([data_class_0_under, target_class_1], ignore_index=True)
            logging.info('Random under-sampling:')
            logging.info(data_test_under.Target.value_counts())

            data_test_under.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test split initiated")
            X_train, X_test, y_train, y_test = train_test_split(data_test_under['Text'],data_test_under['Target'], test_size=0.3, random_state=0)
            
            train_set = pd.concat([X_train, y_train], axis=1)
            test_set = pd.concat([X_test, y_test], axis=1)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, Y_train, X_test, Y_test,_=data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(X_train, Y_train, X_test, Y_test))    