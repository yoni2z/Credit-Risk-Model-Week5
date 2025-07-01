import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load the raw dataset from the specified file path."""
    logger.info(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} not found")
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)

def create_aggregate_features(df):
    """Create aggregate features."""
    logger.info("Creating aggregate features")
    agg_features = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'count', 'std'],
        'TransactionStartTime': ['min', 'max']
    }).reset_index()
    agg_features.columns = ['CustomerId', 'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount', 'FirstTransaction', 'LastTransaction']
    return agg_features

def create_time_features(df):
    """Extract time-based features."""
    logger.info("Extracting time-based features")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True).dt.tz_localize(None)
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    return df

def calculate_rfm(df, snapshot_date):
    """Calculate RFM metrics."""
    logger.info(f"Calculating RFM metrics with snapshot date {snapshot_date}")
    # Ensure TransactionStartTime is timezone-naive
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], format='ISO8601', utc=True).dt.tz_localize(None)
    # Convert snapshot_date to timezone-naive
    snapshot_date = pd.to_datetime(snapshot_date, utc=True).tz_localize(None)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Amount': 'sum'  # Monetary
    }).reset_index()
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm

def cluster_rfm(rfm, n_clusters=3, random_state=42):
    """Cluster customers based on RFM."""
    logger.info(f"Clustering RFM data into {n_clusters} clusters")
    rfm_scaled = StandardScaler().fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    # Identify high-risk cluster (highest Recency, lowest Frequency/Monetary)
    high_risk_cluster = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).idxmax()['Recency']
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    return rfm

def integrate_target(df, rfm):
    """Merge RFM and target variable into main dataset."""
    logger.info("Integrating target variable")
    return df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

def build_preprocessing_pipeline():
    """Build a preprocessing pipeline."""
    logger.info("Building preprocessing pipeline")
    numeric_features = ['Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear', 'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount']
    categorical_features = ['ProductCategory', 'ChannelId', 'PricingStrategy']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def process_data(file_path, snapshot_date='2025-07-01', output_path='data/processed/processed_data.csv'):
    """Main function to process data and save to processed directory."""
    logger.info("Starting data processing")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load and process data
    df = load_data(file_path)
    df = create_time_features(df)
    agg_features = create_aggregate_features(df)
    rfm = calculate_rfm(df, snapshot_date)
    rfm = cluster_rfm(rfm)
    df = integrate_target(df, rfm)
    
    # Merge aggregate features
    df = df.merge(agg_features, on='CustomerId', how='left')
    
    # Apply preprocessing pipeline
    preprocessor = build_preprocessing_pipeline()
    processed_data = preprocessor.fit_transform(df)
    
    # Save the processed DataFrame
    logger.info(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)
    
    logger.info("Data processing completed")
    return processed_data, df, preprocessor

if __name__ == "__main__":
    input_file = "data/raw/data.csv"
    process_data(input_file)

## task 3 and 4 done