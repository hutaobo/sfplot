from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

try:
    import pyarrow.dataset as ds
except ImportError:  # pragma: no cover - optional dependency at runtime
    ds = None


_X_CANDIDATES = ("x_location", "x", "X", "x_um", "global_x")
_Y_CANDIDATES = ("y_location", "y", "Y", "y_um", "global_y")
_FEATURE_CANDIDATES = ("feature_name", "feature", "gene", "FeatureName")


def iter_gene_batches(genes: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    """Yield fixed-size gene batches while preserving the original order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    items = [str(gene) for gene in genes]
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def open_transcript_batch_source(folder: str | Path) -> dict[str, object] | None:
    """
    Return a parquet-backed transcript source when `transcripts.parquet` exists.

    The returned source keeps the pyarrow dataset plus the resolved coordinate and
    feature columns so callers can load only the genes they currently need.
    """
    if ds is None:
        return None

    transcript_path = Path(folder) / "transcripts.parquet"
    if not transcript_path.exists():
        return None

    dataset = ds.dataset(str(transcript_path), format="parquet")
    names = set(dataset.schema.names)
    x_col = _pick_column(names, _X_CANDIDATES)
    y_col = _pick_column(names, _Y_CANDIDATES)
    feature_col = _pick_column(names, _FEATURE_CANDIDATES)

    missing = [name for name, value in (("x", x_col), ("y", y_col), ("feature_name", feature_col)) if value is None]
    if missing:
        raise KeyError(
            "transcripts.parquet is missing required columns: "
            + ", ".join(missing)
        )

    return {
        "dataset": dataset,
        "x_col": x_col,
        "y_col": y_col,
        "feature_col": feature_col,
    }


def load_transcript_batch(
    source: dict[str, object],
    genes: Sequence[str],
) -> pd.DataFrame:
    """
    Load transcripts only for the requested genes.

    Returns a DataFrame with canonical columns `x`, `y`, and `feature_name`.
    """
    dataset = source["dataset"]
    x_col = str(source["x_col"])
    y_col = str(source["y_col"])
    feature_col = str(source["feature_col"])
    requested = [str(gene) for gene in genes]

    if not requested:
        return pd.DataFrame(columns=["x", "y", "feature_name"])

    table = dataset.to_table(
        columns=[x_col, y_col, feature_col],
        filter=ds.field(feature_col).isin(requested),
    )
    if table.num_rows == 0:
        return pd.DataFrame(columns=["x", "y", "feature_name"])

    df = table.to_pandas()
    df = df.rename(columns={x_col: "x", y_col: "y", feature_col: "feature_name"})
    df["feature_name"] = df["feature_name"].astype(str)
    return df.loc[
        ~df["feature_name"].str.contains("NegControl|Unassigned", na=False),
        ["x", "y", "feature_name"],
    ]


def _pick_column(names: set[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None
