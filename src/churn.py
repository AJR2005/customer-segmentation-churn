"""Churn label, behavioural features and model training.

Churn is defined on a time split so the model never sees the future:
features use orders before the cutoff, and a customer is "churned" if they
place no order in the 90 days after it.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

CHURN_WINDOW_DAYS = 90

FEATURES = [
    "recency_days", "frequency", "total_spend", "tenure_days",
    "avg_order_value", "avg_items_per_order", "distinct_products",
    "avg_days_between_orders", "orders_last_90d", "spend_trend",
    "is_uk",
]


def build_features(orders: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """One row per customer, using only orders strictly before `cutoff`."""
    hist = orders[orders["InvoiceDate"] < cutoff]
    inv = hist.groupby(["CustomerID", "InvoiceNo"]).agg(
        date=("InvoiceDate", "min"), spend=("Spend", "sum"), items=("Quantity", "sum")
    ).reset_index()

    g = inv.groupby("CustomerID")
    f = pd.DataFrame({
        "recency_days": (cutoff - g["date"].max()).dt.days,
        "frequency": g["InvoiceNo"].nunique(),
        "total_spend": g["spend"].sum(),
        "tenure_days": (cutoff - g["date"].min()).dt.days,
        "avg_order_value": g["spend"].mean(),
        "avg_items_per_order": g["items"].mean(),
    })
    f["distinct_products"] = hist.groupby("CustomerID")["StockCode"].nunique()

    gaps = inv.sort_values("date").groupby("CustomerID")["date"].apply(lambda d: d.diff().dt.days.mean())
    # One-time buyers have no gap; use tenure as the observed "time without a repeat"
    f["avg_days_between_orders"] = gaps.fillna(f["tenure_days"])

    recent = inv[inv["date"] >= cutoff - pd.Timedelta(days=90)]
    f["orders_last_90d"] = recent.groupby("CustomerID")["InvoiceNo"].nunique().reindex(f.index, fill_value=0)
    # Share of a customer's spend that happened in the last 90 days (1 = all recent, 0 = none)
    f["spend_trend"] = (recent.groupby("CustomerID")["spend"].sum().reindex(f.index, fill_value=0)
                        / f["total_spend"])
    f["is_uk"] = (hist.groupby("CustomerID")["Country"].first() == "United Kingdom").astype(int)
    return f[FEATURES]


def build_labels(orders: pd.DataFrame, customers: pd.Index, cutoff: pd.Timestamp) -> pd.Series:
    """1 if the customer has no order in the churn window after the cutoff."""
    window = orders[(orders["InvoiceDate"] >= cutoff)
                    & (orders["InvoiceDate"] < cutoff + pd.Timedelta(days=CHURN_WINDOW_DAYS))]
    active = set(window["CustomerID"])
    return pd.Series([0 if c in active else 1 for c in customers], index=customers, name="churned")


def candidate_models(seed: int = 42) -> dict:
    log_scale = ColumnTransformer([("log", Pipeline([
        ("log1p", FunctionTransformer(np.log1p)), ("scale", StandardScaler())
    ]), FEATURES[:-1])], remainder="passthrough")
    return {
        "Logistic Regression": Pipeline([("prep", log_scale),
                                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                                class_weight="balanced", random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                                        max_depth=3, subsample=0.8, random_state=seed),
    }


def compare_models(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> pd.DataFrame:
    """5-fold cross-validated ROC-AUC for each candidate."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows = []
    for name, model in candidate_models(seed).items():
        s = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({"model": name, "cv_auc_mean": s.mean(), "cv_auc_std": s.std()})
    return pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)


def lift_table(y_true: pd.Series, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Churn rate and lift by predicted-risk decile (decile 1 = highest risk)."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": proba})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), bins, labels=range(1, bins + 1))
    base = d["y"].mean()
    t = d.groupby("decile", observed=True).agg(customers=("y", "size"), churners=("y", "sum"))
    t["churn_rate"] = t["churners"] / t["customers"]
    t["lift"] = t["churn_rate"] / base
    t["cum_share_of_churners"] = t["churners"].cumsum() / t["churners"].sum()
    return t
