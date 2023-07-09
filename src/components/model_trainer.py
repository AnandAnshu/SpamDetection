import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, X_train, Y_train, X_test, Y_test):
        try:
            logging.info("Creating models and hyper parameters")
    
            models = {
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC()
            }
            params={
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': np.arange(1, 10)
                },
                "SVM":{
                    'C': [100],
                    'gamma':['auto']
                }
            }
            logging.info("Models and hyper parameters passed to evaluate model function")
            model_report: dict=evaluate_models(
                X_train=X_train, 
                y_train=Y_train, 
                X_test=X_test, 
                y_test=Y_test, 
                models=models,
                param=params
            )
            logging.info(model_report)
            #to get best model score from report dict
            best_model_score = max(sorted(model_report.values()))

            #to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]     
            best_model = models[best_model_name]   

            if best_model_score<0.6:
                raise CustomException("No best model found")    

            logging.info("Best model found: {} with accuracy score of: {}".format(best_model_name, best_model_score))

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            score = accuracy_score(Y_test, predicted)
            return score
        except Exception as e:
            raise CustomException(e, sys)