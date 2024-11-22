import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException

# Get project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_object(file_path, obj):
    try:
        # Convert to absolute path if not already
        abs_file_path = os.path.join(ROOT_DIR, file_path)
        dir_path = os.path.dirname(abs_file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(abs_file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        # Convert to absolute path if not already
        abs_file_path = os.path.join(ROOT_DIR, file_path)
        print(f"\nAttempting to load object from: {abs_file_path}")
        
        with open(abs_file_path, "rb") as file_obj:
            loaded_obj = pickle.load(file_obj)
            
        print(f"Successfully loaded object of type: {type(loaded_obj)}")
        return loaded_obj
        
    except Exception as e:
        print(f"\nERROR while loading object: {str(e)}")
        raise CustomException(e, sys)