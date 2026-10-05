import pandas as pd

CLASS_MAP = {
    "benign": 0,
    "malignant": 1,
    "normal": 2,
}

def prepare_csv(csv_path):
    df = pd.read_csv(csv_path)

    if "tipo A" in df.columns:
        df["tipo A"] = df["tipo A"].fillna("").apply(
            lambda x: 1 if x == "rayos X" else 0
        )

    if "tipo B" in df.columns:
        df["tipo B"] = df["tipo B"].fillna("").apply(
            lambda x: 1 if x == "ultrasonido" else 0
        )

    if "class" in df.columns:
        df["label"] = df["class"].map(CLASS_MAP)

    return df
