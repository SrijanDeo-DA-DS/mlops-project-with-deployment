from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import pandas as pd
import yaml
import logging
from typing import Tuple
import mlflow
import mlflow.sklearn

# Set up logging
logging.basicConfig(
    filename='model_training.log',
    filemode='a',  # Set to 'w' if you want to overwrite logs on each run
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_params(params_path: str) -> Tuple[int, int]:
    """
    Load model parameters from a YAML file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        n_estimators = params['model_building']['n_estimators']
        max_depth = params['model_building']['max_depth']
        logging.info(f"Successfully loaded n_estimators and max_depth from {params_path}.")
        return n_estimators, max_depth
    except FileNotFoundError:
        logging.error(f"File {params_path} not found.")
        raise
    except KeyError as e:
        logging.error(f"KeyError: {e} not found in the YAML file.")
        raise
    except yaml.YAMLError:
        logging.error(f"Failed to parse YAML file {params_path}.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading parameters: {e}")
        raise

def train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int, max_depth: int) -> RandomForestRegressor:
    """
    Train the RandomForestRegressor model.
    """
    try:
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        rf.fit(X_train, y_train)
        logging.info("RandomForest model trained successfully.")
        return rf
    except ValueError as e:
        logging.error(f"ValueError while training the model: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during model training: {e}")
        raise

def save_model(model: RandomForestRegressor, file_path: str) -> None:
    """
    Save the trained model to a file.
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f"Model saved successfully to {file_path}.")
    except FileNotFoundError:
        logging.error(f"File path {file_path} not found.")
        raise
    except IOError as e:
        logging.error(f"IOError while saving the model: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving the model: {e}")
        raise

def main():
    try:
        logging.info("Starting the model training pipeline.")
        
        # Load data
        train_data = pd.read_csv('./data/interim/train_interim_transformed.csv')
        test_data = pd.read_csv('./data/interim/test_interim_transformed.csv')
        logging.info("Training and test data loaded successfully.")
        
        # Load parameters
        n_estimators, max_depth = load_params('params.yaml')

        # Prepare data for training
        X_train = train_data.drop(labels=['Time_taken (min)'], axis=1)
        y_train = train_data['Time_taken (min)']  # Ensure this is a Series
        
        # Train the model
        rf = train_model(X_train, y_train, n_estimators, max_depth)
        
        # Save the trained model
        save_model(rf, 'models/model.pkl')

    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logging.error(f"EmptyDataError: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main pipeline: {e}")
        raise

if __name__ == '__main__':
    main()
