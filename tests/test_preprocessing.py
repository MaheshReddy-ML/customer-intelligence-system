import pandas as pd

from src.data.preprocessing import preprocess_data


def test_preprocess_data_removes_invalid_rows_and_converts_dates():
    """Validate cleaning rules for missing customers and invalid transactions."""
    raw_df = pd.DataFrame(
        {
            "CustomerID": [1.0, None, 2.0, 3.0],
            "Quantity": [2, 1, 0, 3],
            "UnitPrice": [10.0, 5.0, 2.0, -1.0],
            "InvoiceDate": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
        }
    )

    cleaned_df = preprocess_data(raw_df)

    assert len(cleaned_df) == 1
    assert cleaned_df.loc[0, "CustomerID"] == 1.0
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["InvoiceDate"])
