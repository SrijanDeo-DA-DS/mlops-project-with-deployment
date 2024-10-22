import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    filename='model_evaluation.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_model(file_path: str) -> RandomForestRegressor:
    """
    Load the trained model from a file.

    Args:
        file_path (str): Path to the model file.

    Returns:
        RandomForestRegressor: Loaded model.
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info(f"Model loaded successfully from {file_path}.")
        return model
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        raise
    except pickle.UnpicklingError:
        logging.error(f"Error unpickling the file {file_path}.")
        raise
    except Exception as e:
        logging.error(f"Error occurred while loading model from {file_path}: {e}")
        raise

def evaluate_model(rf: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate the model using test data.

    Args:
        rf (RandomForestRegressor): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.DataFrame): Test labels.

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics.
    """
    try:
        y_pred = rf.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics_dict = {
            'mean_absolute_error': mae, 
            'mean_squared_error': mse, 
            'r2_score': r2,
            'root_mse': np.sqrt(mse)
        }
        logging.info("Model evaluation metrics computed successfully.")
        return metrics_dict
    except Exception as e:
        logging.error(f"Error occurred while evaluating the model: {e}")
        raise

def save_metrics(metrics: Dict[str, Any], file_path: str) -> None:
    """
    Save the model evaluation metrics to a file.

    Args:
        metrics (Dict[str, Any]): Metrics dictionary to save.
        file_path (str): Path to the file where metrics will be saved.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info(f"Metrics saved successfully to {file_path}.")
    except IOError:
        logging.error(f"Failed to write metrics to {file_path}.")
        raise
    except Exception as e:
        logging.error(f"Error occurred while saving metrics: {e}")
        raise

def main() -> None:
    """
    Main function to load the model, evaluate it, and save the metrics.
    """
    try:
        logging.info("Starting model evaluation pipeline.")
        
        rf = load_model('./models/model.pkl')

        test_data = pd.read_csv('./data/interim/test_interim_transformed.csv')
        logging.info("Test data loaded successfully.")
        
        X_test = test_data.drop(['Time_taken (min)'], axis=1)
        y_test = test_data[['Time_taken (min)']]
        logging.info("Test data split into X_test and y_test.")

        metrics = evaluate_model(rf, X_test, y_test)
        
        save_metrics(metrics, 'reports/metrics.json')
        logging.info("Model evaluation completed and metrics saved.")
    
    except FileNotFoundError as e:
        logging.error(f"File not found during execution: {e}")
        raise
    except pd.errors.EmptyDataError:
        logging.error("Test data file is empty or not found.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during the model evaluation pipeline: {e}")
        raise

if __name__ == '__main__':
    main()
