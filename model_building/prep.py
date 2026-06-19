
import pandas as pd

import os

from sklearn.model_selection import train_test_split

from huggingface_hub import HfApi

# Initialize Hugging Face API

api = HfApi(token=os.getenv("HF_TOKEN"))

# Dataset Path

DATASET_PATH = "hf://datasets/krisna-Labs/visit-with-us-dataset/tourism.csv"

# Read Dataset

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully")

print("Dataset Shape:", df.shape)

# Target and Independent Variables

target = "ProdTaken"

X = df.drop(columns=[target])

y = df[target]

# Categorical and Numerical Variables

categorical_cols = X.select_dtypes(include="object").columns.tolist()

numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

print("\nCategorical Columns:")

print(categorical_cols)

print("\nNumerical Columns:")

print(numerical_cols)

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTrain Shape:", X_train.shape)

print("Test Shape:", X_test.shape)

# Create Processed Folder

os.makedirs(
    "data/processed",
    exist_ok=True
)

# Save Files

X_train.to_csv(

    "data/processed/Xtrain.csv",

    index=False

)

X_test.to_csv(

    "data/processed/Xtest.csv",

    index=False

)

y_train.to_csv(

    "data/processed/ytrain.csv",

    index=False

)

y_test.to_csv(

    "data/processed/ytest.csv",

    index=False

)

print("\nTrain/Test Files Saved Successfully")

# Upload Files to Hugging Face Dataset Repository

api.upload_folder(

    folder_path="data/processed",

    repo_id="krisna-Labs/visit-with-us-dataset",

    repo_type="dataset"

)

print("\nProcessed Files Uploaded Successfully")
