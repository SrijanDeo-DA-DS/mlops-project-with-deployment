import pandas as pd
import numpy as np
import math
import os
import logging
from typing import Any

# Set up logging
logging.basicConfig(
    filename='data_processing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Calculate distance using lat/long
def distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the distance between two geographical points using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: The distance between the two points in kilometers.
    """
    try:
        # Convert latitude and longitude to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        R = 6371.0  # Earth's radius in km
        dist = R * c
        logging.info("Distance calculated successfully.")
        return dist
    except Exception as e:
        logging.error(f"Error occurred during distance calculation: {e}")
        raise

# Calculate Delivery city and Ratings
def cal_ratings(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    Calculate ratings and fill missing values.

    Args:
        df (pd.DataFrame): The dataframe to process.
        col1 (str): Column containing the delivery person IDs.
        col2 (str): Column containing the ratings.

    Returns:
        pd.DataFrame: The dataframe with filled ratings.
    """
    try:
        df['Delivery_city'] = df[col1].str.split('RES', expand=True)[0]
        rating_map = round(df.groupby(col1)[col2].mean(), 1).to_dict()
        df[col2] = df[col2].fillna(df[col1].map(rating_map))
        logging.info("Ratings calculated and missing values filled successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred in cal_ratings: {e}")
        raise

# Extract HH/MM/YYYY from order date
def preprocess_date(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Extract year, month, and day from a date column.

    Args:
        df (pd.DataFrame): The dataframe to process.
        column (str): The column containing the date.

    Returns:
        pd.DataFrame: The dataframe with year, month, and day columns.
    """
    try:
        df[column] = pd.to_datetime(df[column], format='%d-%m-%Y')
        df['year'] = df[column].dt.year
        df['month'] = df[column].dt.month
        df['day'] = df[column].dt.day
        logging.info("Date preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred during date preprocessing: {e}")
        raise

# Preprocess Time ordered
def process_time_ordered(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """
    Preprocess the time ordered column.

    Args:
        df (pd.DataFrame): The dataframe to process.
        time_column (str): The column containing the time ordered.

    Returns:
        pd.DataFrame: The processed dataframe with time columns.
    """
    try:
        df.dropna(subset=[time_column], inplace=True)
        df[time_column] = df[time_column].str.replace('.', ':')

        def extract_time(x: str) -> str:
            try:
                return x.split(':')[0] + ':' + x.split(':')[1][:2]
            except IndexError:
                return '00:00'

        df[time_column] = df[time_column].apply(extract_time)
        df[time_column] = pd.to_datetime(df[time_column], format='%H:%M', errors='coerce')
        df.dropna(subset=[time_column], inplace=True)

        df['TimeOrder_Hour'] = df[time_column].dt.hour
        df['TimeOrder_Hour'] = df['TimeOrder_Hour'].replace(0, 24).astype(int)
        df['TimeOrder_min'] = df[time_column].dt.minute
        logging.info("Time ordered preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred during time ordered preprocessing: {e}")
        raise

# Preprocess Time order picked
def process_time_order_picked(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """
    Preprocess the time order picked column.

    Args:
        df (pd.DataFrame): The dataframe to process.
        time_column (str): The column containing the time order picked.

    Returns:
        pd.DataFrame: The processed dataframe with time columns.
    """
    try:
        df.dropna(subset=[time_column], inplace=True)
        df[time_column] = df[time_column].str.replace('.', ':')

        def extract_time(x: str) -> str:
            try:
                return x.split(':')[0] + ':' + x.split(':')[1][:2]
            except IndexError:
                return '00:00'

        df[time_column] = df[time_column].apply(extract_time)
        df[time_column] = pd.to_datetime(df[time_column], format='%H:%M', errors='coerce')
        df.dropna(subset=[time_column], inplace=True)

        df['Time_Order_picked_Hour'] = df[time_column].dt.hour
        df['Time_Order_picked_Hour'] = df['Time_Order_picked_Hour'].replace(0, 24).astype(int)
        df['Time_Order_picked_min'] = df[time_column].dt.minute
        logging.info("Time order picked preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred during time order picked preprocessing: {e}")
        raise

# Main data preprocessing function
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the entire dataframe by calculating distances, processing dates, and handling time columns.

    Args:
        df (pd.DataFrame): The dataframe to preprocess.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    try:
        df['distance'] = df.apply(lambda row: distance(row['Restaurant_latitude'], row['Restaurant_longitude'], row['Delivery_location_latitude'], row['Delivery_location_longitude']), axis=1)

        df = cal_ratings(df, 'Delivery_person_ID', 'Delivery_person_Ratings')
        df = preprocess_date(df, 'Order_Date')
        df = process_time_ordered(df, 'Time_Orderd')
        df = process_time_order_picked(df, 'Time_Order_picked')

        df.drop(['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude',
                 'year', 'month', 'day', 'TimeOrder_min', 'Time_Order_picked_Hour', 'Time_Order_picked_min',
                 'Time_Orderd', 'Time_Order_picked', 'Delivery_person_ID'], axis=1, inplace=True)

        logging.info("Data preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred during data preprocessing: {e}")
        raise

def main() -> None:
    """
    Main function to load, preprocess, and save the data.
    """
    try:
        logging.info("Starting the data processing pipeline.")
        
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info("Raw data loaded successfully.")

        # Transform the data
        train_processed_data = preprocess_data(train_data)
        test_processed_data = preprocess_data(test_data)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "processed")
        os.makedirs(data_path, exist_ok=True)
        logging.info(f"Processed data directory {data_path} created or exists.")

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logging.info("Processed data saved successfully.")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError:
        logging.error("CSV file is empty or not found.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during the main pipeline: {e}")
        raise

if __name__ == '__main__':
    main()
