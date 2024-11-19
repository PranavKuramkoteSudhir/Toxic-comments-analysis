import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import re
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    max_features = 50000
    min_df = 2
    max_df = 0.95

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [self.clean_text(text) for text in X]
    
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', 'url', text)
        text = re.sub(r'\S+@\S+', 'email', text)
        text = re.sub(r'\d+', 'number', text)
        text = re.sub(r'([!?.])\1+', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            features.append(self.get_text_features(text))
        return pd.DataFrame(features)
    
    def get_text_features(self, text):
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
            'punctuation_ratio': sum(1 for c in text if c in '!?.,') / len(text) if len(text) > 0 else 0,
            'has_url': 1 if 'url' in text else 0,
            'has_email': 1 if 'email' in text else 0
        }

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This function creates the data transformation pipeline
        """
        try:
            # Create the text processing pipeline
            text_pipeline = Pipeline([
                ('preprocessor', TextPreprocessor()),
                ('tfidf', TfidfVectorizer(
                    max_features=self.data_transformation_config.max_features,
                    min_df=self.data_transformation_config.min_df,
                    max_df=self.data_transformation_config.max_df,
                    ngram_range=(1, 2)
                ))
            ])
            
            # Create feature extraction pipeline
            feature_pipeline = Pipeline([
                ('preprocessor', TextPreprocessor()),
                ('feature_extractor', FeatureExtractor())
            ])
            
            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('text_pipeline', text_pipeline, 'comment_text'),
                    ('feature_pipeline', feature_pipeline, 'comment_text')
                ],
                sparse_threshold=0  # Force dense output
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_columns = ['toxic', 'severe_toxic', 'obscene', 
                            'threat', 'insult', 'identity_hate']
            
            # Prepare input features and target features
            input_feature_train_df = train_df[['comment_text']]
            input_feature_test_df = test_df[['comment_text']]
            
            target_feature_train_df = train_df[target_columns]
            target_feature_test_df = test_df[target_columns]
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info(f"Processed training data shape: {input_feature_train_arr.shape}")
            logging.info(f"Processed testing data shape: {input_feature_test_arr.shape}")
            
            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("Saved preprocessing object.")
            
            return (
                input_feature_train_arr,
                input_feature_test_arr,
                target_feature_train_df.values,
                target_feature_test_df.values,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)

def test_transformation_pipeline():
    """
    Function to test the transformation pipeline on a small sample
    """
    try:
        logging.info("Starting transformation pipeline test")
        print("Testing Complete Transformation Pipeline:")
        print("="*50)
        
        # Create instance of DataTransformation
        data_transform = DataTransformation()
        
        # Get the preprocessing object
        preprocessor = data_transform.get_data_transformer_object()
        
        # Read a small sample from the original dataset
        dataset_path = os.path.join('artifacts', 'train.csv')
        sample_data = pd.read_csv(dataset_path).head(3)
        
        # Save sample data as CSV
        sample_train_path = os.path.join('artifacts', 'sample_train.csv')
        sample_test_path = os.path.join('artifacts', 'sample_test.csv')
        sample_data.to_csv(sample_train_path, index=False)
        sample_data.to_csv(sample_test_path, index=False)
        
        logging.info("Created sample data files for testing")
        
        # Test the transformation
        (
            train_arr,
            test_arr,
            train_target_arr,
            test_target_arr,
            preprocessor_path
        ) = data_transform.initiate_data_transformation(sample_train_path, sample_test_path)
        
        print("\nTransformation Results:")
        print(f"Training data shape: {train_arr.shape}")
        print(f"Test data shape: {test_arr.shape}")
        print(f"Training target shape: {train_target_arr.shape}")
        print(f"Test target shape: {test_target_arr.shape}")
        print(f"Preprocessor saved at: {preprocessor_path}")
        
        # Show sample of transformed features
        print("\nSample of transformed features:")
        print(f"Number of features: {train_arr.shape[1]}")
        print("First 5 features of first sample:")
        for i in range(min(5, train_arr.shape[1])):
            print(f"Feature {i}: {train_arr[0][i]}")
        
        logging.info("Transformation pipeline test completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error in testing transformation pipeline: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Test the transformation pipeline
        test_result = test_transformation_pipeline()
        print(f"\nTest completed successfully: {test_result}")
        
        # If test is successful, process the full dataset
        if test_result:
            logging.info("Starting full data transformation")
            obj = DataTransformation()
            train_path = os.path.join('artifacts', 'train.csv')
            test_path = os.path.join('artifacts', 'test.csv')
            
            train_arr, test_arr, train_target_arr, test_target_arr, _ = obj.initiate_data_transformation(
                train_path,
                test_path
            )
            
            logging.info("Data transformation completed successfully")
            logging.info(f"Final training array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")
            
    except Exception as e:
        logging.error("Error in data transformation")
        raise CustomException(e, sys)