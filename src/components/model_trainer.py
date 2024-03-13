import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import customExeption

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifcacts','model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("begining model training")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbours classifier":KNeighborsRegressor(),
                "XGB":XGBRegressor(),
                "cat boost":CatBoostRegressor(verbose=False),
                "adaboost":AdaBoostRegressor()

            }
            hyperparameters = {
            "Random Forest": {
                "n_estimators": [10, 50, 100],
                "max_depth": [None, 5, 10],
                # "min_samples_split": [2, 5, 10],
                # "min_samples_leaf": [1, 2, 4],
                # "max_features": ["auto", "sqrt", "log2"],
                # "criterion": ["mse", "mae"]
            },
            "Decision Tree": {
                "max_depth": [None, 5, 10],
                # "min_samples_split": [2, 5, 10],
                # "min_samples_leaf": [1, 2, 4],
                # "max_features": ["auto", "sqrt", "log2"],
                # "criterion": ["mse", "mae"]
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.5],
                "max_depth": [3, 5, 7],
                # "min_samples_split": [2, 5, 10],
                # "min_samples_leaf": [1, 2, 4],
                # "max_features": ["auto", "sqrt", "log2"]
            },
            "Linear Regression": {
                "fit_intercept": [True, False],
                # "normalize": [True, False]
            },
            "K-Neighbours classifier": {
                "n_neighbors": [5, 10, 20],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
            },
            "XGB": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.5],
                "max_depth": [3, 5, 7],
                # "min_child_weight": [1, 3, 5],
                "gamma": [0.0, 0.1, 0.2],
                # "subsample": [0.6, 0.8, 1.0],
                # "colsample_bytree": [0.6, 0.8, 1.0]
            },
            "cat boost": {
                "iterations": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.5],
                "depth": [3, 5, 7],
                "l2_leaf_reg": [1, 3, 5],
                "border_count": [32, 64, 128]
            },
            "adaboost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.5],
                "loss": ["linear", "square", "exponential"]
            }
        }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }




            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,
                                             x_test=x_test,y_test=y_test,
                                             models=models,params=hyperparameters)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise customExeption("no best model found")
            
            logging.info("best model found on the training ond testing gata")
            save_object(self.model_trainer.trained_model_file_path,best_model)

            pred=best_model.predict(x_test)
            score=r2_score(y_test,pred)
            return score

        except Exception as e:
            raise customExeption(e,sys)