import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import custom exceptions and logger
from src.exception import CustomException
from src.logger import logging

# Import the 'os' module for file path manipulation
import os

# Import custom utility function for saving objects
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # Define the file path for saving the preprocessor object
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for defining the data transformation steps.
        It sets up pipelines for numerical and categorical columns.
        '''
        try:
            # Define the numerical and categorical columns
            numerical_columns = ["reading_score", "writing_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            # Create a pipeline for numerical columns with imputation and scaling
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])

            logging.info(f"Numerical columns: {numerical_columns}")

            # Create a pipeline for categorical columns with imputation, one-hot encoding, and scaling
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("oneHotEncoder", OneHotEncoder()),
                    ("standardScaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")

            # Create a ColumnTransformer that applies the pipelines to specific column sets
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns),
                ]
            )
            return preprocessor

        except Exception as e:
            # Catch and handle exceptions, raising a custom exception
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the training and testing data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train & test data completed.")

            logging.info("Obtaining preprocessing object...")

            # Get the data transformation preprocessor object
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target features for training and testing data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Transform the input features using the preprocessor
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target features as arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save the preprocessing object to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # Catch and handle exceptions, raising a custom exception
            raise CustomException(e, sys)