import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from src.data_processing import create_aggregate_features, create_time_features

def test_create_aggregate_features():
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 200, 150],
        'TransactionStartTime': ['2025-01-01', '2025-02-01', '2025-01-15']
    })
    result = create_aggregate_features(df)
    assert result.shape[0] == 2
    assert result.loc[result['CustomerId'] == 'C1', 'TotalAmount'].iloc[0] == 300
    assert result.loc[result['CustomerId'] == 'C1', 'TransactionCount'].iloc[0] == 2

def test_time_features():
    df = pd.DataFrame({
        'CustomerId': ['C1'],
        'TransactionStartTime': ['2025-01-01 14:30:00']
    })
    result = create_time_features(df)
    assert result['TransactionHour'].iloc[0] == 14
    assert result['TransactionDay'].iloc[0] == 1