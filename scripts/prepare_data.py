from src.data.prepare_data import extract_dataset

if __name__ == "__main__":
    extract_dataset(
        "data/raw/data.zip",
        "data/raw",
    )
