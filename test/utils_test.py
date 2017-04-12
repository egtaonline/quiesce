from egta import utils


def test_random_string():
    string = utils.random_string(5)
    assert len(string) == 5
