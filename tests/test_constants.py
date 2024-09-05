from constants import EVAL_GAME_IDS, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX


def test_eval_game_ids():
    expected_game_ids = ["00.json"]
    assert EVAL_GAME_IDS == expected_game_ids


def test_coordinate_system_constants():
    assert X_MIN == 0
    assert X_MAX == 94.14
    assert Y_MIN == 0
    assert Y_MAX == 50
    assert Z_MIN == 0
    assert Z_MAX == 20
