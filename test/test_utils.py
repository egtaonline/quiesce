"""Tests for utils"""
from egta import utils


def test_random_string():
    """Test that length is preserved"""
    string = utils.random_string(5)
    assert len(string) == 5
