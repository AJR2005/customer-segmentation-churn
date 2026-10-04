"""Load and clean the UCI Online Retail transactions."""
from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/"
    "master/data/retail-data/all/online-retail-dataset.csv"
)
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "online_retail.csv"


def download(path: Path = RAW_PATH) -> Path:
    """Download the raw CSV if it is not already on disk."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(path, index=False)
    return path


def load_clean(path: Path = RAW_PATH) -> tuple[pd.DataFrame, dict]:
    """Return cleaned order lines plus a log of what each cleaning step removed."""
    df = pd.read_csv(download(path), dtype={"CustomerID": str, "InvoiceNo": str})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    log = {"raw_rows": len(df)}

    df = df.dropna(subset=["CustomerID"])
    log["after_dropping_guest_checkouts"] = len(df)

    df = df[~df["InvoiceNo"].str.startswith("C")]  # cancelled orders
    log["after_dropping_cancellations"] = len(df)

    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    # Postage, manual adjustments and bank charges are not customer purchases
    df = df[~df["StockCode"].isin(["POST", "DOT", "M", "BANK CHARGES", "AMAZONFEE", "CRUK", "D", "PADS"])]
    log["after_dropping_non_product_lines"] = len(df)

    df = df.drop_duplicates()
    log["final_rows"] = len(df)

    df["CustomerID"] = df["CustomerID"].str.replace(r"\.0$", "", regex=True)
    df["Spend"] = df["Quantity"] * df["UnitPrice"]
    return df.reset_index(drop=True), log
