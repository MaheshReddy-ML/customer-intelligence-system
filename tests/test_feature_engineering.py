import numpy as np
import pandas as pd

from src.data.feature_engineering import create_customer_features


def test_create_customer_features_builds_expected_customer_level_columns():
    """Validate aggregation output for customer-level modeling features."""
    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 2],
            "Quantity": [2, 3, 5],
            "UnitPrice": [10.0, 20.0, 4.0],
            "InvoiceDate": pd.to_datetime(
                ["2024-01-01", "2024-01-03", "2024-01-04"]
            ),
        }
    )

    features = create_customer_features(df)

    assert list(features.columns) == [
        "CustomerID",
        "Recency",
        "Frequency",
        "MonetaryValue",
        "TotalQuantity",
        "AvgUnitPrice",
    ]
    assert len(features) == 2
    assert features.loc[features["CustomerID"] == 1, "Recency"].item() == 1
    assert np.isclose(
        features.loc[features["CustomerID"] == 1, "Frequency"].item(),
        np.log1p(2),
    )
