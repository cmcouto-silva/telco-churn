import pandas as pd

def read_data(data_path, features=None, target=None):
    if features and target:
        df = pd.read_csv(data_path, index_col='CustomerID', usecols=['CustomerID', *features, target])
    else:
        df = pd.read_csv(data_path, index_col='CustomerID')
    return df
