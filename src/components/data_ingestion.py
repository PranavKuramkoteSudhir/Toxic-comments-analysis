import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        """
        Method to ingest the Jigsaw toxic comment data
        """
        logging.info("Entered the data ingestion method or component")
        
        try:
            # Read the dataset from src/dataset directory
            dataset_path = os.path.join('src', 'dataset', 'train.csv')
            df = pd.read_csv(dataset_path)
            logging.info('Read the dataset as dataframe')
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Perform train-test split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df['toxic']  # Stratify based on toxic label
            )
            
            logging.info("Train test split initiated")
            
            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.error("Exception occurred in Data Ingestion stage")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Train data path: {train_data}, Test data path: {test_data}")
    except Exception as e:
        logging.error("Error in data ingestion")
        raise CustomException(e, sys)