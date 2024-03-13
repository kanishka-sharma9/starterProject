import os
import sys
import numpy as np 
import pandas as pd
from src.exception import customExeption
import dill
from sklearn.model_selection import GridSearchCV
# from src.components.model_trainer import r2_score
from sklearn.metrics import r2_score
def save_object(file_path ,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file:
            dill.dump(obj,file)

    except Exception as e:
        raise customExeption(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            p=params[list(models.keys())[i]]

            gs=GridSearchCV(estimator=model,param_grid=p,scoring='r2',cv=4)
            gs.fit(x_train,y_train)

            # model=gs.best_estimator_
            
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            
            
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_pred_score=r2_score(y_train,y_train_pred)
            test_pred_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_pred_score

        return report
    except Exception as e:
        raise customExeption(e,sys)
    