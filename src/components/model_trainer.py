import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Define a configuration class for the model trainer
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# Create a class for the Model Trainer
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Log that we are splitting training and testing input data
            logging.info("Splitting training and testing input data")
            
            # Split the input data into features (X) and target labels (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define a dictionary of regression models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Evaluate the performance of each model on the training and testing data
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # Find the best model's score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            # Find the name of the best model from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            # Get the best model instance
            best_model = models[best_model_name]
            logging.info(f"Best model is ---> {best_model}")

            # If the best model's score is below a threshold, raise a custom exception
            if best_model_score < 0.6:
                raise CustomException("No best model found..")
            
            logging.info(f"Best found model on both training and testing dataset is {best_model}")

            # Save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions using the best model and calculate R-squared
            predicted = best_model.predict(X_test)
            r2_squared = r2_score(y_test, predicted)
            
            return r2_squared

        except Exception as e:
            raise CustomException(e,sys)