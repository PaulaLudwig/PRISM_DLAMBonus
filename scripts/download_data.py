from pathlib import Path
 
from huggingface_hub import hf_hub_download
 
REPO_ID = "AIML-TUDA/dlam-ts-project-data-2026"
REPO_TYPE = "dataset"
 
FILES = [
    "train.csv",
    "validation_input.csv",
    "forecast_index_validation.csv",
    "metadata.json",
]
 
DATA_DIR = Path(__file__).parent.parent / "data"
 
 
def main():
    DATA_DIR.mkdir(exist_ok=True)
    for filename in FILES:
        print(f"Downloading {filename} ...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type=REPO_TYPE,
            local_dir=DATA_DIR,
        )
        print(f"  -> saved to {local_path}")
    print("Done.")
 
 
if __name__ == "__main__":
    main()