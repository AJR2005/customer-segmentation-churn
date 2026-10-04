"""RFM scoring, rule-based marketing segments and K-Means clusters."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def build_rfm(orders: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Recency (days since last order), Frequency (orders) and Spend per customer."""
    rfm = orders.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: (snapshot - d.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Spend=("Spend", "sum"),
        Country=("Country", "first"),
    )
    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Quintile scores 1-5 (5 = best) for R, F and M."""
    out = rfm.copy()
    out["R"] = pd.qcut(out["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    # rank() breaks ties so qcut can form 5 equal-sized bins on skewed counts
    out["F"] = pd.qcut(out["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M"] = pd.qcut(out["Spend"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["FM"] = ((out["F"] + out["M"]) / 2).round().astype(int)
    out["Segment"] = [_segment(r, fm) for r, fm in zip(out["R"], out["FM"])]
    return out


def _segment(r: int, fm: int) -> str:
    """Standard R x FM grid used by CRM teams."""
    if r >= 4 and fm >= 4:
        return "Champions"
    if r >= 3 and fm >= 3:
        return "Loyal Customers"
    if r >= 4 and fm <= 2:
        return "New / Promising"
    if r == 3 and fm <= 2:
        return "Needs Attention"
    if r <= 2 and fm >= 4:
        return "Can't Lose Them"
    if r <= 2 and fm == 3:
        return "At Risk"
    return "Hibernating"


def _scaled(rfm: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Spend"]]))


def kmeans_diagnostics(rfm: pd.DataFrame, k_range=range(2, 9), seed: int = 42) -> pd.DataFrame:
    """Inertia (elbow) and silhouette for each k, on log-scaled RFM."""
    X = _scaled(rfm)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def fit_kmeans(rfm: pd.DataFrame, k: int, seed: int = 42) -> pd.Series:
    """Cluster labels, renumbered so cluster 0 has the lowest median Recency (most active)."""
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(_scaled(rfm))
    order = pd.Series(rfm["Recency"].values).groupby(labels).median().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    return pd.Series([remap[l] for l in labels], index=rfm.index, name="Cluster")
