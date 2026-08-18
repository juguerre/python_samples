import pandas as pd
import pytest
from pandas import DataFrame

from samples.merge_snapshot_in_history import (
    SAMPLE_HISTORY_DF,
    SAMPLE_SNAPSHOT_DF,
    merge_snapshot_in_history,
)


def test_merge_snapshot_in_history_sample() -> None:
    """Test merging sample snapshot into sample history."""
    result = merge_snapshot_in_history(
        snapshot=SAMPLE_SNAPSHOT_DF,
        history=SAMPLE_HISTORY_DF,
        key_columns=["id"],
        descriptive_columns=["company", "email"],
        bussines_date_str="2026-01-03",
    )

    # 1. Row count should be 5:
    # - 1 closed historical (username3: 2026-01-01 -> 2026-01-02)
    # - 1 updated in-place (username1)
    # - 1 closed/deleted (username4: valid_to = 2026-01-02)
    # - 2 newly inserted active (username2, username3)
    assert len(result) == 5

    # 2. Check updated in-place row (username1)
    u1 = result[(result["id"] == "username1") & (result["valid_to"].isna())].iloc[0]
    assert u1["email"] == "email1_mod"
    assert u1["description"] == "description1_mod"
    assert u1["valid_from"] == "2026-01-01"
    assert u1["updated_at"] == "2026-01-03"
    assert u1["sk_id"] == "username1,2026-01-01"

    # 3. Check closed/deleted row (username4)
    u4 = result[result["id"] == "username4"].iloc[0]
    assert u4["valid_to"] == "2026-01-02"
    assert u4["sk_id"] == "username4,2026-01-01,2026-01-02"

    # 4. Check newly inserted active rows
    u2 = result[result["id"] == "username2"].iloc[0]
    assert u2["valid_from"] == "2026-01-03"
    assert pd.isna(u2["valid_to"])
    assert u2["updated_at"] == "2026-01-03"
    assert u2["sk_id"] == "username2,2026-01-03"

    u3_active = result[(result["id"] == "username3") & (result["valid_to"].isna())].iloc[0]
    assert u3_active["valid_from"] == "2026-01-03"
    assert u3_active["email"] == "email3_mod"


def test_merge_snapshot_missing_columns_validation() -> None:
    """Test validation errors when required columns are missing."""
    invalid_history = DataFrame({"id": ["user1"]})
    with pytest.raises(ValueError, match="History dataframe must contain all required columns"):
        merge_snapshot_in_history(
            snapshot=SAMPLE_SNAPSHOT_DF,
            history=invalid_history,
            key_columns=["id"],
            descriptive_columns=["company"],
            bussines_date_str="2026-01-03",
        )
