import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Online Retail transactions for modeling.

    Args:
        df: Raw Online Retail DataFrame.

    Returns:
        Cleaned transaction DataFrame.
    """
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.reset_index(drop=True)

    return df
