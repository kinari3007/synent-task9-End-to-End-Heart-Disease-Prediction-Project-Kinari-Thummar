import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Decision Tree':       DecisionTreeClassifier(max_depth=5),
                'Random Forest':       RandomForestClassifier(n_estimators=100)
            }

            report = evaluate_models(X_train, y_train, X_test, y_test, models)
            logging.info(f"Model scores: {report}")

            best_name  = max(report, key=report.get)
            best_score = report[best_name]
            best_model = models[best_name]
            best_model.fit(X_train, y_train)

            save_object(self.config.model_path, best_model)
            logging.info(f"Best model: {best_name} | Accuracy: {best_score:.4f}")

            return best_score, best_name
        except Exception as e:
            raise CustomException(e, sys)