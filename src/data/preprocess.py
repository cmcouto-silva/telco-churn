import pandas as pd
from src.utils import read_data

def process_data(raw_data_path, processed_data_path):

    df_raw = read_data(raw_data_path)

    df_processed = df_raw.copy()
    df_processed['Total Charges'] = pd.to_numeric(df_processed['Total Charges'], errors='coerce')
    df_processed = df_processed.dropna(subset=['Total Charges'])
    df_processed['Churn'] = df_processed['Churn Value'].astype(int)
    df_processed.to_csv(processed_data_path)
    