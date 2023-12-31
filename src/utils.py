import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """
    Save an object to a file using dill serialization.

    Args:
        file_path (str): The file path where the object will be saved.
        obj: The object to be saved.

    Raises:
        CustomException: If an error occurs while saving the object.

    """
    try:
        dir_path = os.path.dirname(file_path)

        # Ensure that the directory path exists or create it if it doesn't
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    """
    Evaluate a set of machine learning models using R-squared (R2) score.

    Args:
        X_train (pandas.DataFrame): The training data features.
        y_train (pandas.Series): The training data target variable.
        X_test (pandas.DataFrame): The test data features.
        y_test (pandas.Series): The test data target variable.
        models (dict): A dictionary of machine learning models to evaluate.

    Returns:
        dict: A dictionary where keys are model names and values are R-squared (R2) scores.

    Raises:
        CustomException: If an error occurs during model evaluation.

    """
    try:
        # Create an empty dictionary to store model evaluation reports.
        report = {}
        # Iterate through a list of machine learning models.
        for i in range(len(list(models))):
            
            # Get the hyperparameter grid for grid search for the current model.
            model = list(models.values())[i]

            # Get the hyperparameter grid for grid search for the current model.
            parameters = param[list(models.keys())[i]]

            # Initialize GridSearchCV with the model and hyperparameter grid, using 3-fold cross-validation.
            gs = GridSearchCV(model,parameters,cv=3)
            
            # Perform grid search to find the best hyperparameters for the model using training data.
            gs.fit(X_train,y_train)

            # Set the model's hyperparameters to the best parameters found during grid search.
            model.set_params(**gs.best_params_)

            # Fit the model to the training data with the best hyperparameters.    
            model.fit(X_train, y_train)

            # Make predictions on the training and test data using the trained model.
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R-squared (R2) score for both training and test sets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the model's evaluation results in the report dictionary, using the model's name as the key.
            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)

# Function to open the file_path in the readback(rb) mode and loading the pkl file  
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)