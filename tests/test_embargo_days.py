import pytest
from configs.base_config import ProjectConfig

def test_bad_split_rejected():
    with pytest.raises(ValueError):
        ProjectConfig(TRAIN_SPLIT=0.9, VAL_SPLIT=0.2)

def test_embargo_follows_lookback():
    assert ProjectConfig(LOOKBACK_WINDOW=30).EMBARGO_DAYS == 30
    