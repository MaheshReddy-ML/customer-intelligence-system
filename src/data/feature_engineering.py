import numpy as np
import pandas as pd


def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer-level behavioral features from retail transactions.

    Args:
        df: Cleaned online retail transaction DataFrame.

    Returns:
        Customer-level feature DataFrame.
    """
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    reference_date = df["InvoiceDate"].max()
    customer_features = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "CustomerID": "count",
        "TotalPrice": "sum",
        "Quantity": "sum",
        "UnitPrice": "mean"
    })
    customer_features.columns = [
        "Recency",
        "Frequency",
        "MonetaryValue",
        "TotalQuantity",
        "AvgUnitPrice"
    ]
    customer_features = customer_features.reset_index()
    customer_features["Frequency"] = np.log1p(customer_features["Frequency"])
    customer_features["MonetaryValue"] = np.log1p(
        customer_features["MonetaryValue"]
    )
    customer_features["TotalQuantity"] = np.log1p(
        customer_features["TotalQuantity"]
    )

    return customer_features
