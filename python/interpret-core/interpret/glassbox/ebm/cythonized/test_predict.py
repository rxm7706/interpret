import pytest
from .predict import sum


def test_predict():
    a, b = 1, 2
    result = sum(a, b)
    assert result == 3
