import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self):
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("Train/test data loaded for transformation")

            target = 'target'
            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]
            X_test  = test_df.drop(columns=[target])
            y_test  = test_df[target]

            preprocessor = self.get_preprocessor()
            X_train_arr  = preprocessor.fit_transform(X_train)
            X_test_arr   = preprocessor.transform(X_test)

            save_object(self.config.preprocessor_path, preprocessor)
            logging.info("Preprocessor saved to artifacts/")

            return X_train_arr, X_test_arr, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)