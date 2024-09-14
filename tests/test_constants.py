from nba_tracking_data_commons.constants import EVAL_GAME_IDS, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX


def test_eval_game_ids():
    expected_first_game_id = "0021500033.json"
    assert EVAL_GAME_IDS[0] == expected_first_game_id


def test_coordinate_system_constants():
    assert X_MIN == 0
    assert X_MAX == 94.14
    assert Y_MIN == 0
    assert Y_MAX == 50
    assert Z_MIN == 0
    assert Z_MAX == 20
