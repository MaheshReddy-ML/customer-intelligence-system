from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


RAW_DATA_PATH = Path("data/raw")
DATA_FILE = RAW_DATA_PATH / "online_retail.csv"


def fetch_online_retail_data(dataset_id: int = 352) -> pd.DataFrame:
    """Load the Online Retail dataset from disk or download it from UCI.

    Args:
        dataset_id: UCI repository dataset identifier.

    Returns:
        Raw Online Retail DataFrame.
    """
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        print("Loading dataset from local CSV...")
        return pd.read_csv(DATA_FILE)

    print("Downloading dataset from UCI repository...")

    online_retail = fetch_ucirepo(id=dataset_id)

    df = online_retail.data.features

    df.to_csv(DATA_FILE, index=False)

    print(f"Dataset saved locally at: {DATA_FILE}")

    return df


if __name__ == "__main__":
    df = fetch_online_retail_data()
    print(df.head())
