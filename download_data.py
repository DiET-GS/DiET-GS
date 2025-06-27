from huggingface_hub import snapshot_download
import os
import shutil

def download_from_huggingface():
    snapshot_download(
        repo_id="onandon/DiET-GS",
        repo_type="dataset",
        local_dir="data",
        local_dir_use_symlinks=False
    )

if __name__ == "__main__":
    print("INFO : Downloading the preprocessed data from huggingface...")
    download_from_huggingface()