import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from data_ingestion import DataIngestion
from data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            
            # Convert lists to numpy arrays if needed
            X_train = np.array(train_array)  # Training features
            X_test = np.array(test_array[1])  # Test features
            y_train = np.array(test_array[0])  # Training labels
            y_test = np.array(test_array[2])  # Test labels
            
            logging.info(f"Training data shape: {X_train.shape}")
            logging.info(f"Test data shape: {X_test.shape}")
            logging.info(f"Training target shape: {y_train.shape}")
            logging.info(f"Test target shape: {y_test.shape}")

            # Define models suitable for multi-label text classification
            models = {
                "Logistic Regression": MultiOutputClassifier(
                    LogisticRegression(max_iter=1000, n_jobs=-1)
                ),
                "Naive Bayes": MultiOutputClassifier(
                    MultinomialNB()
                ),
                "Linear SVM": MultiOutputClassifier(
                    LinearSVC(max_iter=1000)
                ),
                "Random Forest": MultiOutputClassifier(
                    RandomForestClassifier(n_estimators=100, n_jobs=-1)
                ),
                "Gradient Boosting": MultiOutputClassifier(
                    GradientBoostingClassifier()
                )
            }

            def evaluate_models(X_train, y_train, X_test, y_test, models):
                model_performances = {}
                
                for model_name, model in models.items():
                    try:
                        logging.info(f"Training {model_name}")
                        model.fit(X_train, y_train)
                        
                        # Predict probabilities for ROC AUC calculation
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics for each toxic category
                        category_scores = {}
                        for i, category in enumerate(self.toxic_columns):
                            try:
                                accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
                                # For SVM, we won't have predict_proba
                                if hasattr(model, "predict_proba"):
                                    y_pred_proba = model.predict_proba(X_test)
                                    auc = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
                                else:
                                    auc = roc_auc_score(y_test[:, i], y_pred[:, i])
                                
                                category_scores[category] = {
                                    'accuracy': accuracy,
                                    'auc': auc
                                }
                                
                                logging.info(f"{model_name} - {category}:")
                                logging.info(f"Accuracy: {accuracy:.4f}")
                                logging.info(f"AUC: {auc:.4f}")
                                logging.info(f"\n{classification_report(y_test[:, i], y_pred[:, i])}")
                                
                            except Exception as e:
                                logging.warning(f"Error calculating metrics for {category}: {str(e)}")
                        
                        # Calculate mean scores across all categories
                        mean_accuracy = np.mean([scores['accuracy'] for scores in category_scores.values()])
                        mean_auc = np.mean([scores['auc'] for scores in category_scores.values()])
                        
                        model_performances[model_name] = {
                            'model': model,
                            'mean_accuracy': mean_accuracy,
                            'mean_auc': mean_auc,
                            'category_scores': category_scores
                        }
                        
                        logging.info(f"\n{model_name} Mean Scores:")
                        logging.info(f"Mean Accuracy: {mean_accuracy:.4f}")
                        logging.info(f"Mean AUC: {mean_auc:.4f}\n")
                        
                    except Exception as e:
                        logging.error(f"Error training {model_name}: {str(e)}")
                        continue
                
                return model_performances

            # Evaluate all models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            if not model_report:
                raise CustomException("No models were successfully trained", sys)

            # Get the best model based on mean AUC
            best_model_name = max(model_report.items(), 
                                key=lambda x: x[1]['mean_auc'])[0]
            best_model_info = model_report[best_model_name]
            best_model = best_model_info['model']
            best_score = best_model_info['mean_auc']

            logging.info(f"Best performing model: {best_model_name}")
            logging.info(f"Best model mean AUC: {best_score:.4f}")

            if best_score < 0.75:
                raise CustomException("No model achieved acceptable performance (AUC < 0.75)", sys)

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Save detailed results
            results = []
            for model_name, perf in model_report.items():
                for category, scores in perf['category_scores'].items():
                    results.append({
                        'Model': model_name,
                        'Category': category,
                        'Accuracy': scores['accuracy'],
                        'AUC': scores['auc']
                    })
            
            results_df = pd.DataFrame(results)
            results_file = os.path.join("artifacts", "model_results.csv")
            results_df.to_csv(results_file, index=False)
            logging.info(f"Saved detailed results to {results_file}")

            return best_score

        except Exception as e:
            raise CustomException(e, sys)

def test_model_trainer(sample_size=1000):
    try:
        logging.info(f"Testing with sample size: {sample_size}")
        
        data_ingestion = DataIngestion()
        data_transformation = DataTransformation()
        model_trainer = ModelTrainer()
        
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        # Get transformed data
        train_arr, test_arr, train_target_arr, test_target_arr, _ = \
            data_transformation.initiate_data_transformation(train_path, test_path)
        
        if sample_size:
            train_arr = train_arr[:sample_size]
            train_target_arr = train_target_arr[:sample_size]
            test_arr = test_arr[:sample_size]
            test_target_arr = test_target_arr[:sample_size]
        
        # Pack arrays correctly
        test_array = [train_target_arr, test_arr, test_target_arr]  # Changed the order and included test features
        
        model_score = model_trainer.initiate_model_trainer(train_arr, test_array)
        logging.info(f"Test completed with score: {model_score}")
        return True
        
    except Exception as e:
        logging.error(f"Error in testing model trainer: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        
            
        logging.info("Processing full dataset")
        test_model_trainer(sample_size=None)
            
    except Exception as e:
        logging.error("Error in model training pipeline")
        raise CustomException(e, sys)
