import hashlib
import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://github.com/adendek/DeepGate-FaceRec/releases/download/data-v1/faces.zip"
SHA256_SUM = "6bd700f9eebe5dad3561bcad79fd949b6db25b3759974faef16d54fe5f1cdefd"
DEST_ZIP = Path("data/faces.zip")
EXTRACT_DIR = Path("data/")


def setup_dataset() -> None:
    """Download and extract dataset for the release."""
    if Path("data/split/train").exists():
        print("Dataset already unpacked.")
        return

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    if not DEST_ZIP.exists():
        print(f"Downloading faces.zip from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DEST_ZIP)
        print("Download complete.")

    # Check sum
    sha256 = hashlib.sha256()
    with open(DEST_ZIP, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    if sha256.hexdigest() != SHA256_SUM:
        raise ValueError("Downloaded file has incorrect SHA256 checksum.")
    print("Checksum verified successfully:", sha256.hexdigest())

    print("Extracting dataset...")
    with zipfile.ZipFile(DEST_ZIP, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Dataset ready in data/")


if __name__ == "__main__":
    setup_dataset()
