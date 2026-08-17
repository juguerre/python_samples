# merge_snapshot_in_history.py
# Script to merge content snapshot in history using SCD type 2 pattern
from datetime import datetime, timedelta

import pandas as pd
from pandas import DataFrame, isna

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
            "username1,company1,email1",
            "username4,company4,email4",
            "username3,company3,email3",
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
    """Merge a generic content snapshot into a generic history using SCD type 2 pattern.

    Key columns are the columns that identify the unique row in the snapshot and will be used to
    generate hash_diff column.
    Surrogate key is the column that identify the unique row in the history has the key columns plus
    the timestamp columns: valid_from, valid_to and updated_at.

    :param snapshot: The snapshot to merge into the history.
    :param history: The history to merge the snapshot into.
    :param key_columns: The columns that identify the unique row in the snapshot.
    :param descriptive_columns: The columns that describe the row in the snapshot.
    :param bussines_date_str: The business date to use for the valid_from and valid_to columns.
    :return: The merged history.
    """
    # 1. check history dataframe for required columns: valid_from, valid_to, updated_at and sk_id
    required_scd2_columns = ["valid_from", "valid_to", "updated_at", "sk_id", "hash_diff"]
    if not all(col in history.columns for col in required_scd2_columns):
        raise ValueError(
            f"History dataframe must contain all required columns: {required_scd2_columns}"
        )
    # 2. check history and snapshot for key and descriptive columns
    columns_to_check = key_columns + descriptive_columns
    if not all(col in snapshot.columns for col in columns_to_check):
        raise ValueError(
            f"Snapshot dataframe must contain all key and descriptive columns: {key_columns}"
        )
    if not all(col in history.columns for col in columns_to_check):
        raise ValueError(
            f"History dataframe must contain all key and descriptive columns: {key_columns}"
        )
    current_history = history[history["valid_to"].isnull()]
    sk_columns = key_columns + ["valid_from", "valid_to"]

    # 3. generate hash_diff column
    snapshot["hash_diff"] = snapshot[key_columns + descriptive_columns].apply(
        lambda row: ",".join(row.values), axis=1
    )

    # 4. Find snapshot rows with key_columns not present in current history (valid_to == null)
    new_rows = _get_new_rows(
        current_history=current_history,
        snapshot=snapshot,
        bussines_date_str=bussines_date_str,
        key_columns=key_columns,
        sk_columns=sk_columns,
    )

    # 5. Find updated rows in snapshot (rows present in both current history and snapshot but with
    # different hash_diff for the same key_columns)
    # join snapshot with current history on key_columns
    updated_rows = _get_updated_rows(
        current_history=current_history,
        snapshot=snapshot,
        bussines_date_str=bussines_date_str,
        key_columns=key_columns,
    )

    # 6. Find deleted rows in snapshot (rows present in current history but not in snapshot)
    deleted_rows = _get_deleted_rows(
        current_history=current_history,
        snapshot=snapshot,
        bussines_date_str=bussines_date_str,
        key_columns=key_columns,
    )

    # 7. Merge history with new, updated and deleted rows
    history = _get_merged_history(
        history=history,
        new_rows=new_rows,
        updated_rows=updated_rows,
        deleted_rows=deleted_rows,
        sk_columns=sk_columns,
    )

    return history


def _get_merged_history(
    history: DataFrame,
    new_rows: DataFrame,
    updated_rows: DataFrame,
    deleted_rows: DataFrame,
    sk_columns: list[str],
) -> DataFrame:
    history = pd.concat([history, new_rows], ignore_index=True)

    # 8. Merge updated (update with updated_at)
    # replace rows in history with updated rows
    history = history[~history[["sk_id"]].isin(updated_rows[["sk_id"]]).all(axis=1)]
    history = pd.concat([history, updated_rows], ignore_index=True)

    # 9. Merge deleted (update with valid_to)
    history = history[~history[["sk_id"]].isin(deleted_rows[["sk_id"]]).all(axis=1)]
    deleted_rows["sk_id"] = deleted_rows[sk_columns].apply(
        _generate_sk_id, args=(sk_columns,), axis=1
    )
    history = pd.concat([history, deleted_rows], ignore_index=True)
    return history


def _get_deleted_rows(
    current_history: DataFrame,
    snapshot: DataFrame,
    bussines_date_str: str,
    key_columns: list[str],
) -> DataFrame:
    deleted_rows = current_history[
        ~current_history[key_columns].isin(snapshot[key_columns]).all(axis=1)
    ]
    deleted_rows = deleted_rows.reset_index(drop=True)
    valid_to = (
        datetime.strptime(bussines_date_str, "%Y-%m-%d").date() - timedelta(days=1)
    ).isoformat()
    deleted_rows["valid_to"] = valid_to
    return deleted_rows


def _get_updated_rows(
    current_history: DataFrame,
    snapshot: DataFrame,
    bussines_date_str: str,
    key_columns: list[str],
) -> DataFrame:
    merged_snapshot = snapshot.merge(
        current_history, on=key_columns, suffixes=("_x", "_y"), how="inner"
    )
    updated_rows = merged_snapshot[merged_snapshot["hash_diff_x"] != merged_snapshot["hash_diff_y"]]
    updated_rows = updated_rows.reset_index(drop=True)
    # drop columns with suffix _y and rename columns with suffix _x
    scd2_cols = ["valid_from", "valid_to", "updated_at"]
    history_cols_to_drop = [
        col for col in updated_rows.columns if col.endswith("_y") and col not in scd2_cols
    ]
    updated_rows = updated_rows.drop(columns=history_cols_to_drop)
    updated_rows = updated_rows.rename(
        columns={
            f"{col}": col.rsplit("_", 1)[-2] for col in updated_rows.columns if col.endswith("_x")
        }
    )
    updated_rows["updated_at"] = bussines_date_str
    updated_rows["valid_to"] = pd.NA
    return updated_rows


def _get_new_rows(
    current_history: DataFrame,
    snapshot: DataFrame,
    key_columns: list[str],
    bussines_date_str: str,
    sk_columns: list[str],
) -> DataFrame:
    new_rows = snapshot[~snapshot[key_columns].isin(current_history[key_columns]).all(axis=1)]
    new_rows = new_rows.reset_index(drop=True)
    new_rows["valid_from"] = bussines_date_str
    new_rows["valid_to"] = pd.NA
    new_rows["updated_at"] = bussines_date_str
    new_rows["sk_id"] = new_rows[sk_columns].apply(_generate_sk_id, args=(sk_columns,), axis=1)
    return new_rows


def _generate_sk_id(row: pd.Series, sk_columns: list[str]) -> str:
    non_null_values = [row[c] for c in sk_columns if not isna(row[c])]
    return ",".join(non_null_values)


if __name__ == "__main__":
    histo = merge_snapshot_in_history(
        SAMPLE_SNAPSHOT_DF,
        SAMPLE_HISTORY_DF,
        ["id"],
        ["company", "email"],
        "2026-01-03",
    )
    # TODO print sample of histo df without using print func
    # print(histo.to_string())
