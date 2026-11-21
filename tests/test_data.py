import pandas as pd
import pytest

from powerplant import config
from powerplant.data import (
    DataContractError,
    drop_exact_duplicates,
    glue_sheets,
    load_raw,
    load_sheets,
    split_holdout,
    validate_contract,
)

COLUMNS = ["AT", "V", "AP", "RH", "PE"]
GOOD_ROWS = [
    (14.96, 41.76, 1024.07, 73.17, 463.26),
    (25.18, 62.96, 1020.04, 59.08, 444.37),
    (5.11, 39.40, 1012.16, 92.14, 488.56),
]


def make_frame(rows: list[tuple[float, ...]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture(scope="module")
def clean_data() -> pd.DataFrame:
    return drop_exact_duplicates(load_raw())


def test_load_raw_reads_all_rows_with_expected_columns():
    df = load_raw()

    assert list(df.columns) == COLUMNS
    assert len(df) == 9568


def test_load_sheets_finds_five_copies_of_the_same_size():
    sheets = load_sheets()

    assert len(sheets) == 5
    assert {len(frame) for frame in sheets.values()} == {9568}


def test_glued_sheets_are_mostly_repeated_rows():
    glued = glue_sheets(load_sheets())

    assert len(glued) == 47840
    assert len(glued.round(4).drop_duplicates()) == 9527


def test_validate_contract_accepts_good_data():
    df = make_frame(GOOD_ROWS)

    assert validate_contract(df).equals(df)


def test_validate_contract_allows_humidity_sensor_overshoot():
    df = make_frame([(14.96, 41.76, 1024.07, 100.16, 463.26)])

    assert len(validate_contract(df)) == 1


def test_validate_contract_rejects_missing_column():
    df = make_frame(GOOD_ROWS).drop(columns=["V"])

    with pytest.raises(DataContractError, match="V"):
        validate_contract(df)


def test_validate_contract_rejects_missing_values():
    df = make_frame(GOOD_ROWS + [(None, 41.76, 1024.07, 73.17, 463.26)])

    with pytest.raises(DataContractError, match="пропуск"):
        validate_contract(df)


def test_validate_contract_rejects_values_out_of_range():
    df = make_frame(GOOD_ROWS + [(14.96, 41.76, 1024.07, 150.0, 463.26)])

    with pytest.raises(DataContractError, match="RH"):
        validate_contract(df)


def test_drop_exact_duplicates_returns_new_frame():
    df = make_frame(GOOD_ROWS + [GOOD_ROWS[0]])

    clean = drop_exact_duplicates(df)

    assert len(clean) == 3
    assert len(df) == 4


def test_real_data_has_41_duplicates(clean_data):
    assert len(clean_data) == 9527


def test_split_holdout_is_deterministic(clean_data):
    first = split_holdout(clean_data, stratify_bins=10)
    second = split_holdout(clean_data, stratify_bins=10)

    assert first.test.equals(second.test)


def test_split_holdout_respects_test_size(clean_data):
    split = split_holdout(clean_data, test_size=0.2)

    assert len(split.test) == pytest.approx(0.2 * len(clean_data), abs=1)
    assert len(split.train) + len(split.test) == len(clean_data)


def test_split_holdout_has_no_shared_rows(clean_data):
    split = split_holdout(clean_data, stratify_bins=10)

    shared = split.train.merge(split.test, how="inner", on=COLUMNS)

    assert shared.empty


def test_stratified_split_keeps_target_distribution(clean_data):
    split = split_holdout(clean_data, stratify_bins=10)

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    train_q = split.train[config.TARGET].quantile(quantiles).to_numpy()
    test_q = split.test[config.TARGET].quantile(quantiles).to_numpy()

    assert abs(train_q - test_q).max() < 1.5
