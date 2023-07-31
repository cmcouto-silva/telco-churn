import pickle
from prince import FAMD
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer


def split_data(df, features, target):
    X,y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=2023)
    return X_train, X_test, y_train, y_test


def train_cluster(X, y, features=None):
    cluster_pipeline = make_pipeline(
        make_column_transformer([FAMD(n_components=20), features]),
        KMeans(n_clusters=4, n_init='auto', random_state=2023)
    )
    cluster_pipeline.fit(X)
    return cluster_pipeline


def train_churn(X, y, numeric_features, categorical_features):

    preprocessor = ColumnTransformer([
        ('PowerTransformer', PowerTransformer(), numeric_features),
        ('onehot', OneHotEncoder(drop='if_binary', sparse_output=False), categorical_features),
        ('onehot_clst', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['cluster'])
    ]).set_output(transform='pandas')

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(C=0.03151363, max_iter=1_000, class_weight='balanced'))
    ])

    model_pipeline.fit(X, y)
    return model_pipeline


def train_models(df, numeric_features, categorical_features, target):
    """Train cluster and churn models"""
    # Split dataset
    features = numeric_features + categorical_features
    X_train, X_test, y_train, y_test = split_data(df, features, target)
    
    # Train cluster model
    cluster_model = train_cluster(X_train, y_train, features)

    # Add cluster predictons
    X_train['cluster'] = cluster_model.predict(X_train)
    X_test['cluster'] = cluster_model.predict(X_test)
    
    # Train churn model with clusters
    churn_model = train_churn(X_train, y_train, numeric_features, categorical_features)

    # Return fitted models
    return cluster_model, churn_model


def save_models(cluster_model, churn_model, cluster_model_path, churn_model_path):
    """Serialize and save models"""
    # Save cluster model
    with open(cluster_model_path, 'wb') as file:
        pickle.dump(cluster_model, file)

    # Save churn model
    with open(churn_model_path, 'wb') as file:
        pickle.dump(churn_model, file)
