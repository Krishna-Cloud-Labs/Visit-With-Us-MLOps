
import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="deployment",
    repo_id="krisna-Labs/visit-with-us-mlops",
    repo_type="space"
)

print("Hosting files uploaded successfully")
