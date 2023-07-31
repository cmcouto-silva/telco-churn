RAW_DATA_PATH = 'data/raw/customer_churn.csv'
PROCESSED_DATA_PATH = 'data/processed/customer_churn.csv'

CLUSTER_MODEL_PATH = 'models/cluster_pipeline.pkl'
CHURN_MODEL_PATH = 'models/prediction_pipeline.pkl'

NUMERIC_FEATURES = [
    'Tenure Months',
    'Monthly Charges',
    'Total Charges',
    'CLTV'
]

CATEGORICAL_FEATURES = [
    'Senior Citizen',
    'Partner',
    'Dependents',
    'Multiple Lines',
    'Internet Service',
    'Online Security',
    'Online Backup',
    'Device Protection',
    'Tech Support',
    'Streaming TV',
    'Streaming Movies',
    'Contract',
    'Paperless Billing',
    'Payment Method'
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = 'Churn'
