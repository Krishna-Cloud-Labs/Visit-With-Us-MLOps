
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "krisna-Labs/visit-with-us-dataset"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type
    )
    print(f"Dataset '{repo_id}' already exists. Using it.")

except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )

    print(f"Dataset '{repo_id}' created.")

api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Dataset uploaded successfully.")
