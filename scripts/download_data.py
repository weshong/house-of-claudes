"""Download competition data from Kaggle."""

import subprocess
import zipfile
from pathlib import Path

from marchmadness.config import DATA_DIR


def download():
    """Download and extract competition data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading data from Kaggle...")
    result = subprocess.run(
        ["kaggle", "competitions", "download",
         "-c", "march-machine-learning-mania-2026",
         "-p", str(DATA_DIR)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    print(result.stdout)

    # Extract any zip files
    for zip_path in DATA_DIR.glob("*.zip"):
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()
        print(f"  Extracted and removed {zip_path.name}")

    # List extracted files
    csvs = list(DATA_DIR.glob("*.csv"))
    print(f"\nExtracted {len(csvs)} CSV files:")
    for csv in sorted(csvs):
        print(f"  {csv.name}")

    return True


if __name__ == "__main__":
    download()
