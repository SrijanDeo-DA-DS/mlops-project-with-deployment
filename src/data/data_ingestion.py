import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml
import logging
from typing import Tuple

# Set up logging
logging.basicConfig(
    filename='app.log', 
    filemode='a', 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

def load_params(params_path: str) -> float:
    """
    Load parameters from a YAML file.

    Args:
        params_path (str): Path to the YAML file.

    Returns:
        float: The test size parameter.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        logging.info(f"Successfully loaded test_size from {params_path}.")
        return test_size
    except FileNotFoundError:
        logging.error(f"File {params_path} not found.")
        raise
    except KeyError:
        logging.error("Key 'test_size' not found in the YAML file.")
        raise
    except yaml.YAMLError:
        logging.error(f"Failed to parse YAML file {params_path}.")
        raise

def read_data(url: str) -> pd.DataFrame:
    """
    Read data from a CSV file.

    Args:
        url (str): URL of the CSV file.

    Returns:
        pd.DataFrame: The dataframe with the loaded data.
    """
    try:
        df = pd.read_csv(url)
        logging.info(f"Successfully read data from {url}.")
        return df
    except pd.errors.EmptyDataError:
        logging.error("No data found in the CSV file.")
        raise
    except pd.errors.ParserError:
        logging.error("Error parsing the CSV file.")
        raise
    except Exception as e:
        logging.error(f"Failed to read the CSV file from {url}. Error: {e}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the dataframe by dropping the 'ID' column if present.

    Args:
        df (pd.DataFrame): The dataframe to process.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    try:
        if 'ID' in df.columns:
            df.drop('ID', axis=1, inplace=True)
            logging.info("Successfully dropped 'ID' column from the dataframe.")
        else:
            logging.warning("'ID' column not found in the dataframe.")
        return df
    except Exception as e:
        logging.error(f"Failed to process the data. Error: {e}")
        raise

def save_data(raw_data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """
    Save the train and test data to CSV files.

    Args:
        raw_data_path (str): The directory path to save the files.
        train_data (pd.DataFrame): The training data to save.
        test_data (pd.DataFrame): The test data to save.
    """
    try:
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.info(f"Successfully saved train and test data to {raw_data_path}.")
    except PermissionError:
        logging.error(f"Permission denied when saving files to {raw_data_path}.")
        raise
    except Exception as e:
        logging.error(f"Failed to save data. Error: {e}")
        raise

def main() -> None:
    """
    Main function to execute the data processing pipeline.
    """
    try:
        logging.info("Starting the data processing pipeline.")
        
        # Read the data
        df = read_data('https://raw.githubusercontent.com/Shivan118/New-Machine-Learning-Modular-Coding-projecs/refs/heads/main/Data/finalTrain.csv')
        df = process_data(df)
        
        # Load parameters
        test_size = load_params('params.yaml')
        logging.info(f"Loaded test size: {test_size}.")
        
        # Split the data into train and test sets
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        logging.info("Successfully split the data into train and test sets.")
        
        # Create raw data path
        data_path = './data'
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        logging.info(f"Raw data path {raw_data_path} created.")
        
        # Save the data
        save_data(raw_data_path, train_data, test_data)
    
    except Exception as e:
        logging.error(f"An error occurred during the execution: {e}")
        raise

if __name__ == '__main__':
    main()
