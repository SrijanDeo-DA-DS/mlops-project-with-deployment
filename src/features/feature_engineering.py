from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import os
import logging
from typing import List, Any, Union, Tuple
from sklearn.base import TransformerMixin

# Set up logging
logging.basicConfig(
    filename='preprocessing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def create_preprocessor(
    numerical_columns: List[str], 
    categorical_columns: List[str], 
    ordinal_columns: List[str], 
    ordinal_categories: List[List[str]]
) -> ColumnTransformer:
    """
    Create a ColumnTransformer that applies different preprocessing pipelines to 
    numerical, categorical, and ordinal columns.

    Args:
        numerical_columns (List[str]): List of numerical column names.
        categorical_columns (List[str]): List of categorical column names.
        ordinal_columns (List[str]): List of ordinal column names.
        ordinal_categories (List[List[str]]): List of lists of categories for each ordinal column.

    Returns:
        ColumnTransformer: A ColumnTransformer with preprocessing pipelines.
    """
    try:
        # Numerical pipeline
        numerical_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler(with_mean=False))
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ('scaler', StandardScaler(with_mean=False))
        ])
        
        # Ordinal pipeline
        ordinal_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(categories=ordinal_categories)),
            ('scaler', StandardScaler(with_mean=False))
        ])
        
        # Column transformer
        preprocessor = ColumnTransformer([
            ('numerical_pipeline', numerical_pipeline, numerical_columns),
            ('categorical_pipeline', categorical_pipeline, categorical_columns),
            ('ordinal_pipeline', ordinal_pipeline, ordinal_columns)
        ],)
        
        logging.info("Preprocessor created successfully.")
        return preprocessor
    except Exception as e:
        logging.error(f"Error creating preprocessor: {e}")
        raise

def transform_data(
    preprocessor: ColumnTransformer, 
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    numerical_columns: List[str], 
    categorical_columns: List[str], 
    ordinal_columns: List[str], 
    ordinal_categories: List[List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform the train and test data using the provided preprocessor.

    Args:
        preprocessor (ColumnTransformer): A ColumnTransformer for transforming the data.
        df_train (pd.DataFrame): The training dataframe.
        df_test (pd.DataFrame): The testing dataframe.
        numerical_columns (List[str]): List of numerical column names.
        categorical_columns (List[str]): List of categorical column names.
        ordinal_columns (List[str]): List of ordinal column names.
        ordinal_categories (List[List[str]]): List of lists of categories for each ordinal column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Transformed training and testing dataframes.
    """
    try:
        # Fit the preprocessor on the train data and transform both train and test data
        df_train_transformed = preprocessor.fit_transform(df_train)
        df_test_transformed = preprocessor.transform(df_test)

        # Get feature names for the OneHotEncoder
        ohe_columns = list(preprocessor.named_transformers_['categorical_pipeline']['onehot'].get_feature_names_out(categorical_columns))

        # Combine all column names
        all_columns = numerical_columns + ohe_columns + ordinal_columns

        # Convert the NumPy arrays back to DataFrames with appropriate column names
        df_train_transformed = pd.DataFrame(df_train_transformed, columns=all_columns)
        df_test_transformed = pd.DataFrame(df_test_transformed, columns=all_columns)

        logging.info("Data transformed successfully.")
        return df_train_transformed, df_test_transformed
    except Exception as e:
        logging.error(f"Error during data transformation: {e}")
        raise

def main() -> None:
    """
    Main function to load data, create a preprocessor, transform the data, and save it.
    """
    try:
        logging.info("Starting the data transformation pipeline.")

        # Fetch the data from data/processed
        train_data = pd.read_csv('./data/processed/train_processed.csv')
        test_data = pd.read_csv('./data/processed/test_processed.csv')
        logging.info("Raw data loaded successfully.")

        # Separating the target column
        target_col_train = train_data['Time_taken (min)']
        target_col_test = test_data['Time_taken (min)']

        train_data.drop(['Time_taken (min)'], axis=1, inplace=True)
        test_data.drop(['Time_taken (min)'], axis=1, inplace=True)

        Road_traffic_density = ['Low', 'Medium', 'High', 'Jam']
        Weather_conditions = ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Sandstorms', 'Stormy']

        categorical_columns = ['Type_of_order', 'Type_of_vehicle', 'Festival', 'City', 'Delivery_city']
        ordinal_columns = ['Road_traffic_density', 'Weather_conditions']
        ordinal_categories = [Road_traffic_density, Weather_conditions]
        numerical_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition', 'multiple_deliveries',
                             'TimeOrder_Hour', 'distance']

        preprocessor = create_preprocessor(numerical_columns, categorical_columns, ordinal_columns, ordinal_categories)

        train_data_transformed, test_data_transformed = transform_data(preprocessor, train_data, test_data, numerical_columns,
                                                                       categorical_columns, ordinal_columns, ordinal_categories)

        # Concatenating the target column back
        train_data_transformed = pd.concat([train_data_transformed, target_col_train], axis=1)
        test_data_transformed = pd.concat([test_data_transformed, target_col_test], axis=1)

        # Store the data inside data/interim
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        logging.info(f"Interim data directory {data_path} created.")

        train_data_transformed.to_csv(os.path.join(data_path, "train_interim_transformed.csv"), index=False)
        test_data_transformed.to_csv(os.path.join(data_path, "test_interim_transformed.csv"), index=False)
        logging.info("Transformed data saved successfully.")
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
