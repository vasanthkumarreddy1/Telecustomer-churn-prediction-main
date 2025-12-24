import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Loading model and preprocessor objects")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("Transforming data")
            data_scaled = preprocessor.transform(features)
            
            print("Making predictions")
            # Use predict_proba to get confidence scores
            probabilities = model.predict_proba(data_scaled)
            
            # Get the prediction (0 or 1) by finding the class with the highest probability
            prediction = np.argmax(probabilities, axis=1)
            
            # Get the confidence score for the predicted class
            confidence = np.max(probabilities, axis=1)

            return prediction, confidence

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        gender: str,
        Partner: str,
        Dependents: str,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        OnlineBackup: str,
        DeviceProtection: str,
        TechSupport: str,
        StreamingTV: str,
        StreamingMovies: str,

        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        tenure: int,
        MonthlyCharges: float,
        TotalCharges: float
    ):
        self.gender = gender
        self.Partner = Partner
        self.Dependents = Dependents
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.tenure = tenure
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "tenure": [self.tenure],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)