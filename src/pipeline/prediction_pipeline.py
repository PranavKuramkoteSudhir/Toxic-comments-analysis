import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import TextPreprocessor, FeatureExtractor

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Loading model and preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("Model and preprocessor loaded successfully")
            
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            
            prediction_labels = ['Toxic', 'Severely Toxic', 'Obscene', 
                               'Threat', 'Insult', 'Identity Hate']
            
            results = {}
            for idx, label in enumerate(prediction_labels):
                results[label] = "Yes" if predictions[0][idx] > 0.5 else "No"
                
            return results
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, comment_text: str):
        self.comment_text = comment_text

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "comment_text": [self.comment_text]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)