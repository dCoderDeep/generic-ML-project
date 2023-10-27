import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score

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

def evaluate_models(X_train, y_train, X_test, y_test, models):
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
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R-squared (R2) score for both training and test sets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
