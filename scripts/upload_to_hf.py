from huggingface_hub import HfApi, create_repo

HF_USERNAME = "bravo1311"   # replace
MODEL_REPO = f"{HF_USERNAME}/flow-matching-landing-policy"
DATASET_REPO = f"{HF_USERNAME}/flow-matching-landing-episodes"
api = HfApi()

# --- Model repo ---
create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj="checkpoints/v1/model.pt",
    path_in_repo="model.pt",
    repo_id=MODEL_REPO,
    repo_type="model",
)
api.upload_file(
    path_or_fileobj="checkpoints/v1/README.md",
    path_in_repo="README.md",
    repo_id=MODEL_REPO,
    repo_type="model",
)

# --- Dataset repo (synthetic PD demonstrations) ---
create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path="data/episodes",
    repo_id=DATASET_REPO,
    repo_type="dataset",
)
api.upload_folder(
    folder_path="model/flow_matching_v1",
    repo_id=MODEL_REPO,
    repo_type="model",
    path_in_repo="src",   # lands in a src/ subfolder on the repo
)

print("Upload complete.")
print(f"Model: https://huggingface.co/{MODEL_REPO}")
print(f"Dataset: https://huggingface.co/datasets/{DATASET_REPO}")