import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'heart.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            df = pd.read_csv(self.config.raw_data_path)
            logging.info(f"Dataset loaded: {df.shape}")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(self.config.train_data_path, index=False)
            test.to_csv(self.config.test_data_path, index=False)
            logging.info("Train/test split complete")

            return self.config.train_data_path, self.config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)