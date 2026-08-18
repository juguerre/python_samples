# merge_snapshot_in_history.py
# Script to merge content snapshot in history using SCD type 2 pattern with in-place updates
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, cast

import pandas as pd
from pandas import DataFrame, Series, isna


def _compute_hash_diff(values: Sequence[Any]) -> str:
    """Return a SHA-256 hex-digest for a list of values, normalizing nulls to empty string."""
    normalized = ",".join(str(v) if not isna(v) else "" for v in values)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _compute_vectorized_hash_diff(df: DataFrame, columns: Sequence[str]) -> list[str]:
    """Return SHA-256 hex-digests for specified columns in a fast vectorized manner."""
    if not columns or df.empty:
        return [""] * len(df)
    series_list: Series[str] = cast(Series, [df[col].fillna("").astype(str) for col in columns])
    joined_series = series_list[0]
    for s in series_list[1:]:
        joined_series = joined_series + "," + s
    return [hashlib.sha256(val.encode("utf-8")).hexdigest() for val in joined_series]


def _generate_vectorized_sk_id(df: DataFrame, sk_columns: Sequence[str]) -> pd.Series:
    """Generate surrogate key IDs by vectorially concatenating non-null sk_columns."""
    if df.empty or not sk_columns:
        return pd.Series([], dtype=str, index=df.index)
    result: Series[str] = df[sk_columns[0]].astype(str)
    for col in sk_columns[1:]:
        if col in df.columns:
            has_val = ~df[col].isna()
            col_str = df[col].astype(str)
            result: Series[str] = result.where(~has_val, result + "," + col_str)  # ty:ignore[unsupported-operator]
    return result


SAMPLE_SNAPSHOT_DF = DataFrame(
    {
        "id": ["username1", "username2", "username3"],
        "company": ["company1", "company2", "company3"],
        "email": ["email1_mod", "email2", "email3_mod"],
        "description": ["description1_mod", "description2", "description3_mod"],
    }
)

SAMPLE_HISTORY_DF = DataFrame(
    data={
        "sk_id": [
            "username1,2026-01-01",
            "username4,2026-01-01",
            "username3,2026-01-01,2026-01-02",
        ],
        "valid_from": ["2026-01-01", "2026-01-01", "2026-01-01"],
        "valid_to": [pd.NA, pd.NA, "2026-01-02"],
        "updated_at": ["2026-01-02", "2026-01-01", "2026-01-01"],
        "hash_diff": [
            _compute_hash_diff(["username1", "company1", "email1"]),
            _compute_hash_diff(["username4", "company4", "email4"]),
            _compute_hash_diff(["username3", "company3", "email3"]),
        ],
        "id": ["username1", "username4", "username3"],
        "company": ["company1", "company4", "company3"],
        "email": ["email1", "email4", "email3"],
        "description": ["description1", "description4", "description3"],
    }
)


def merge_snapshot_in_history(
    snapshot: DataFrame,
    history: DataFrame,
    key_columns: list[str],
    descriptive_columns: list[str],
    bussines_date_str: str,
) -> DataFrame:
    """Merge a generic content snapshot into history using SCD type 2 pattern.

    - New rows: Inserted with valid_from = bussines_date_str, valid_to = pd.NA.
    - Updated rows: Updated in-place with new attributes and updated_at = bussines_date_str.
    - Deleted rows: Closed with valid_to = bussines_date_str - 1 day and updated sk_id.
    - Historical / Closed rows: Retained unchanged.

    :param snapshot: The snapshot DataFrame to merge.
    :param history: The existing history DataFrame.
    :param key_columns: The columns that uniquely identify a row.
    :param descriptive_columns: The tracked descriptive columns.
    :param bussines_date_str: The business date (YYYY-MM-DD).
    :return: The merged history DataFrame.
    """
    # 1. Validate required columns
    _validate_required_columns(descriptive_columns, history, key_columns, snapshot)

    # 2. Defensive copy and fast vectorized hash computation
    snapshot = snapshot.copy()
    hash_cols = key_columns + descriptive_columns
    snapshot["hash_diff"] = _compute_vectorized_hash_diff(snapshot, hash_cols)

    # 3. Partition history into closed historical records and current active records
    is_active = history["valid_to"].isna()
    closed_history = history[~is_active].copy()
    current_history = history[is_active].copy()

    valid_to_deleted = (
        datetime.strptime(bussines_date_str, "%Y-%m-%d").date() - timedelta(days=1)
    ).isoformat()

    # 4. Single-pass full outer join between current active history and snapshot
    snap_non_key = [c for c in snapshot.columns if c not in key_columns]
    snap_prep = snapshot.rename(columns={c: f"{c}__snap" for c in snap_non_key})

    merged = current_history.merge(
        snap_prep,
        on=key_columns,
        how="outer",
        indicator="_merge_indicator",
    )

    is_deleted = merged["_merge_indicator"] == "left_only"
    is_new = merged["_merge_indicator"] == "right_only"
    is_both = merged["_merge_indicator"] == "both"

    is_updated = is_both & (merged["hash_diff"] != merged["hash_diff__snap"])
    is_unchanged = is_both & (merged["hash_diff"] == merged["hash_diff__snap"])

    # Part A: Unchanged active records
    unchanged_df = merged.loc[is_unchanged, history.columns].copy()

    # Part B: Updated active records (in-place update)
    if is_updated.any():
        updated_df = merged.loc[is_updated].copy()
        for c in snap_non_key:
            updated_df[c] = updated_df[f"{c}__snap"]
        updated_df["updated_at"] = bussines_date_str
        updated_df["valid_to"] = pd.NA
        updated_df = updated_df[history.columns]
    else:
        updated_df = DataFrame(columns=history.columns)

    # Part C: Deleted records (close valid_to and recompute sk_id)
    if is_deleted.any():
        deleted_df = merged.loc[is_deleted, history.columns].copy()
        deleted_df["valid_to"] = valid_to_deleted
        sk_columns = key_columns + ["valid_from", "valid_to"]
        deleted_df["sk_id"] = _generate_vectorized_sk_id(deleted_df, sk_columns)
    else:
        deleted_df = DataFrame(columns=history.columns)

    # Part D: New records (insert active row with valid_from = bussines_date_str)
    if is_new.any():
        snap_cols_in_merged = key_columns + [f"{c}__snap" for c in snap_non_key]
        new_df: DataFrame = merged.loc[is_new, snap_cols_in_merged].copy()  # ty:ignore[invalid-assignment]
        snap_rename_back = {f"{c}__snap": c for c in snap_non_key}
        new_df = new_df.rename(columns=snap_rename_back)
        new_df["valid_from"] = bussines_date_str
        new_df["valid_to"] = pd.NA
        new_df["updated_at"] = bussines_date_str
        sk_columns = key_columns + ["valid_from"]
        new_df["sk_id"] = _generate_vectorized_sk_id(new_df, sk_columns)

        # Align with history schema
        for c in history.columns:
            if c not in new_df.columns:
                new_df[c] = pd.NA
        new_df = new_df[history.columns]
    else:
        new_df = DataFrame(columns=history.columns)

    # 5. Single concatenation preserving original history column ordering
    result = pd.concat(
        [closed_history, unchanged_df, updated_df, deleted_df, new_df],
        ignore_index=True,
    )
    return result[history.columns]


def _validate_required_columns(
    descriptive_columns: list[str], history: DataFrame, key_columns: list[str], snapshot: DataFrame
):
    required_scd2_columns = ["valid_from", "valid_to", "updated_at", "sk_id", "hash_diff"]
    missing_history_scd = [col for col in required_scd2_columns if col not in history.columns]
    if missing_history_scd:
        raise ValueError(
            f"History dataframe must contain all required columns: {missing_history_scd}"
        )

    columns_to_check = key_columns + descriptive_columns
    missing_snap = [col for col in columns_to_check if col not in snapshot.columns]
    if missing_snap:
        raise ValueError(
            f"Snapshot dataframe must contain all key and descriptive columns: {missing_snap}"
        )

    missing_hist = [col for col in columns_to_check if col not in history.columns]
    if missing_hist:
        raise ValueError(
            f"History dataframe must contain all key and descriptive columns: {missing_hist}"
        )


if __name__ == "__main__":
    histo = merge_snapshot_in_history(
        SAMPLE_SNAPSHOT_DF,
        SAMPLE_HISTORY_DF,
        ["id"],
        ["company", "email"],
        "2026-01-03",
    )
    # print(histo.to_string())
