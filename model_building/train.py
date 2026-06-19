
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# MLFLOW CONFIGURATION


mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment(
    "Visit-With-Us-Package-Prediction-Experiment"
)


# HUGGING FACE API

api = HfApi(token=os.getenv("HF_TOKEN"))


# LOAD DATA


Xtrain_path = "https://huggingface.co/datasets/krisna-Labs/visit-with-us-dataset/resolve/main/Xtrain.csv"

Xtest_path = "https://huggingface.co/datasets/krisna-Labs/visit-with-us-dataset/resolve/main/Xtest.csv"

ytrain_path = "https://huggingface.co/datasets/krisna-Labs/visit-with-us-dataset/resolve/main/ytrain.csv"

ytest_path = "https://huggingface.co/datasets/krisna-Labs/visit-with-us-dataset/resolve/main/ytest.csv"

X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)

y_train = pd.read_csv(ytrain_path)
y_test = pd.read_csv(ytest_path)

y_train = y_train.squeeze()
y_test = y_test.squeeze()

print("Data Loaded Successfully")

# DROP ID COLUMNS

drop_cols = ["Unnamed: 0", "CustomerID"]

X_train = X_train.drop(
    columns=drop_cols,
    errors="ignore"
)

X_test = X_test.drop(
    columns=drop_cols,
    errors="ignore"
)


# FEATURE TYPES

cat_cols = X_train.select_dtypes(
    include="object"
).columns.tolist()

num_cols = X_train.select_dtypes(
    exclude="object"
).columns.tolist()

print("\nCategorical Columns:")
print(cat_cols)

print("\nNumerical Columns:")
print(num_cols)


# PREPROCESSOR

preprocessor = make_column_transformer(

    (
        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]),
        num_cols
    ),

    (
        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]),
        cat_cols
    )
)


# MODEL

rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf)
])


# HYPERPARAMETERS

params = {

    "model__n_estimators": [100, 200],

    "model__max_depth": [5, 10, None],

    "model__min_samples_split": [2, 5],
    
    "model__min_samples_leaf": [1, 2, 4]
}


# TRAINING

with mlflow.start_run():

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(
        X_train,
        y_train
    )

    best_model = grid.best_estimator_

    print("\nBest Parameters:")
    print(grid.best_params_)

    print("\nBest CV Score:")
    print(grid.best_score_)

    
    # LOG PARAMETERS

    mlflow.log_params(
        grid.best_params_
    )

    mlflow.log_metric(
        "best_cv_score",
        grid.best_score_
    )

  
    # PREDICTIONS

    train_pred = best_model.predict(
        X_train
    )

    test_pred = best_model.predict(
        X_test
    )

    
    # METRICS

    train_acc = accuracy_score(
        y_train,
        train_pred
    )

    test_acc = accuracy_score(
        y_test,
        test_pred
    )

    precision = precision_score(
        y_test,
        test_pred
    )

    recall = recall_score(
        y_test,
        test_pred
    )

    f1 = f1_score(
        y_test,
        test_pred
    )

    print("\nTrain Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    mlflow.log_metric(
        "train_accuracy",
        train_acc
    )

    mlflow.log_metric(
        "test_accuracy",
        test_acc
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    
    # SAVE MODEL

    os.makedirs(
        "Visit-With-Us-MLOps/model",
        exist_ok=True
    )

    model_path = (
        "Visit-With-Us-MLOps/model/"
        "tourism_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="tourism_model",
    serialization_format="pickle"
)

    print("\nModel Saved Successfully")


# REGISTER MODEL TO HUGGING FACE

repo_id = "krisna-Labs/visit-with-us-mlops"
repo_type = "model"

try:

    api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type
    )

    print("\nModel Repository Exists")

except RepositoryNotFoundError:

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        exist_ok=True
    )

    print("\nModel Repository Created")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="tourism_model.pkl",
    repo_id=repo_id,
    repo_type=repo_type
)

print("\nModel Uploaded Successfully")
