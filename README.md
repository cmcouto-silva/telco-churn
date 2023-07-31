# A hybrid approach for predicting and preventing churn

As part of my MBA in Data Science & Analytics, I proposed a hybrid approach using unsupervised and supervised models for predicting customer churn from a telecom company.

### Model training:

The unsupervised pipeline includes dimensionality reduction using Factor analysis of mixed data (FAMD) followed by K-means clustering, which allowed us to develop better strategies based on the cluster patterns.

The supervised model pipeline includes Yeo-Johnson transformation to the numerical features, one-hot encoding to the categorical features, and weighted logistic regression for churn classification.

All code was developed using best MLOps practices, open-source pipelines, and proper validation, therefore avoiding data leakage.

### Model deployment:

The model was deployed using a web app for interactive "what if" or "batch" predictions and a REST API with proper documentation.

**Links:**
- Web app
  - [Web app deployed on the Streamlit cloud](https://cmcouto-silva-telco-churn-app-stream-project-description-18l8js.streamlit.app/)
  - [Docker image repository](https://hub.docker.com/repository/docker/cmcoutosilva/telco-churn-streamlit-app/general)
  - [Code repository](https://github.com/cmcouto-silva/telco-churn-app)
- REST API
  - [REST API deployed on GCP App Engine](https://model-deployment-376218.rj.r.appspot.com/docs) (framework: FastAPI)
  - [Docker image repository](https://hub.docker.com/repository/docker/cmcoutosilva/telco-churn-rest-api)
  - [Code repository](https://github.com/cmcouto-silva/telco-churn-api)
