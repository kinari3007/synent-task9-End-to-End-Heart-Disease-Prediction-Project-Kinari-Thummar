import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def predict(self, features):
        try:
            preprocessor = load_object('artifacts/preprocessor.pkl')
            model        = load_object('artifacts/model.pkl')
            scaled       = preprocessor.transform(features)
            pred         = model.predict(scaled)
            proba        = model.predict_proba(scaled)[0][1]
            return pred[0], proba
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, age, sex, cp, trestbps, chol, fbs,
                 restecg, thalach, exang, oldpeak, slope, ca, thal):
        self.age=age;         self.sex=sex;       self.cp=cp
        self.trestbps=trestbps; self.chol=chol;   self.fbs=fbs
        self.restecg=restecg; self.thalach=thalach; self.exang=exang
        self.oldpeak=oldpeak; self.slope=slope;   self.ca=ca
        self.thal=thal

    def get_data_as_dataframe(self):
        return pd.DataFrame([{
            'age':self.age,       'sex':self.sex,       'cp':self.cp,
            'trestbps':self.trestbps, 'chol':self.chol, 'fbs':self.fbs,
            'restecg':self.restecg, 'thalach':self.thalach, 'exang':self.exang,
            'oldpeak':self.oldpeak, 'slope':self.slope, 'ca':self.ca,
            'thal':self.thal
        }])