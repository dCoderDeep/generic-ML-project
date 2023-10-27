import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

# Define a data class to store data ingestion configuration parameters.
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# DataIngestion class responsible for handling data ingestion.
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        # Log that the data ingestion method or component is entered.
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from a CSV file located at 'notebook\data\stud.csv'.
            df = pd.read_csv('notebook\data\stud.csv')
            # Log that the dataset is read as a dataframe.

            # Create the necessary directories for the data paths if they don't exist.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to the specified path.
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Log that the train-test split is initiated.
            logging.info("Train test split initiated")

            # Split the dataset into training and test sets.
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training set to the specified path.
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Save the test set to the specified path.
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Log that data ingestion is completed.
            logging.info("Ingestion of the data is completed")

            # Return the paths to the training and test data.
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # If an exception occurs, raise a CustomException and provide the error message and sys information.
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Create an instance of the DataIngestion class and initiate the data ingestion process.
    obj = DataIngestion()
    # Initiate data ingestion to obtain training and testing data.
    train_data, test_data = obj.initiate_data_ingestion()

    # Create an instance of the DataTransformation class.
    data_transformation = DataTransformation()
    # Initiate data transformation on the training and testing data.
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Create an instance of the ModelTrainer class.
    modeltrainer = ModelTrainer()
    # Initiate the model training process and print the result.
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
