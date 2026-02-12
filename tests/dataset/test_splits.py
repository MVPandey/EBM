import pandas as pd

from ebm.dataset.splits import split_dataset


def _make_df(n: int) -> pd.DataFrame:
    puzzle = '0' * 81
    solution = '1' * 81
    return pd.DataFrame({'puzzle': [puzzle] * n, 'solution': [solution] * n})


def test_split_sizes():
    df = _make_df(1000)
    train, val, test = split_dataset(df, val_size=100, test_size=100)
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100


def test_split_deterministic():
    df = _make_df(500)
    train1, val1, test1 = split_dataset(df, val_size=50, test_size=50)
    train2, val2, test2 = split_dataset(df, val_size=50, test_size=50)
    assert train1.index.tolist() == train2.index.tolist()
    assert val1.index.tolist() == val2.index.tolist()
    assert test1.index.tolist() == test2.index.tolist()


def test_no_overlap():
    df = _make_df(200)
    # Give rows unique IDs so we can check disjointness
    df['id'] = range(len(df))
    train, val, test = split_dataset(df, val_size=30, test_size=30)
    train_ids = set(train['id'])
    val_ids = set(val['id'])
    test_ids = set(test['id'])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids | val_ids | test_ids) == 200
