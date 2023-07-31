from src import config
from src.data import preprocess
from src.utils import read_data
from src.models.train import train_models, save_models

# Pre-process & load data
preprocess.process_data(config.RAW_DATA_PATH, config.PROCESSED_DATA_PATH)
df = read_data(config.PROCESSED_DATA_PATH)

# Train & serialize models
cluster_model, churn_model = train_models(df, config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES, config.TARGET)
save_models(cluster_model, churn_model, config.CLUSTER_MODEL_PATH, config.CHURN_MODEL_PATH)
