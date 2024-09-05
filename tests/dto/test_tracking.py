from dto.tracking import Coordinate, Frame, Event


def test_coordinate_initialization():
    coord = Coordinate(x=1.0, y=2.0, z=3.0)
    assert coord.x == 1.0
    assert coord.y == 2.0
    assert coord.z == 3.0


def test_coordinate_to_row():
    coord = Coordinate(x=1.0, y=2.0, z=3.0)
    assert coord.to_row() == [1.0, 2.0, 3.0]


def test_frame_initialization():
    coord1 = Coordinate(x=1.0, y=2.0, z=3.0)
    coord2 = Coordinate(x=4.0, y=5.0, z=6.0)
    players = {1: coord1, 2: coord2}
    frame = Frame(ball=coord1, players=players)

    assert frame.ball == coord1
    assert frame.players == players


def test_event_initialization():
    coord1 = Coordinate(x=1.0, y=2.0, z=3.0)
    coord2 = Coordinate(x=4.0, y=5.0, z=6.0)
    players = {1: coord1, 2: coord2}
    frame = Frame(ball=coord1, players=players)

    event = Event(
        game_id="game123",
        event_id=1,
        period=2,
        game_clock_start=0.0,
        game_clock_end=10.0,
        wall_clock_start=100.0,
        wall_clock_end=110.0,
        team_0_id=0,
        team_1_id=1,
        team_0_players=[1],
        team_1_players=[2],
        frames=[frame],
    )

    assert event.game_id == "game123"
    assert event.event_id == 1
    assert event.period == 2
    assert event.game_clock_start == 0.0
    assert event.game_clock_end == 10.0
    assert event.wall_clock_start == 100.0
    assert event.wall_clock_end == 110.0
    assert event.team_0_id == 0
    assert event.team_1_id == 1
    assert event.team_0_players == [1]
    assert event.team_1_players == [2]
    assert event.frames == [frame]


def test_coordinate_from_dict():
    data = {"x": 1.1, "y": 2.2, "z": 3.3}
    coord = Coordinate.from_dict(data)
    assert coord == Coordinate(x=1.1, y=2.2, z=3.3)


def test_frame_from_dict():
    data = {
        "ball": {"x": 1.1, "y": 2.2, "z": 3.3},
        "players": {"1": {"x": 4.4, "y": 5.5, "z": 6.6}, "2": {"x": 7.7, "y": 8.8, "z": 9.9}},
    }
    frame = Frame.from_dict(data)
    expected_frame = Frame(
        ball=Coordinate(x=1.1, y=2.2, z=3.3),
        players={1: Coordinate(x=4.4, y=5.5, z=6.6), 2: Coordinate(x=7.7, y=8.8, z=9.9)},
    )
    assert frame == expected_frame


def test_event_from_dict():
    data = {
        "game_id": "game123",
        "event_id": 1,
        "period": 2,
        "game_clock_start": 0.0,
        "game_clock_end": 90.0,
        "wall_clock_start": 1657012345.0,
        "wall_clock_end": 1657012435.0,
        "team_0_id": 10,
        "team_1_id": 20,
        "team_0_players": [1, 2, 3],
        "team_1_players": [4, 5, 6],
        "frames": [
            {
                "ball": {"x": 1.1, "y": 2.2, "z": 3.3},
                "players": {"1": {"x": 4.4, "y": 5.5, "z": 6.6}, "2": {"x": 7.7, "y": 8.8, "z": 9.9}},
            }
        ],
    }
    event = Event.from_dict(data)
    expected_event = Event(
        game_id="game123",
        event_id=1,
        period=2,
        game_clock_start=0.0,
        game_clock_end=90.0,
        wall_clock_start=1657012345.0,
        wall_clock_end=1657012435.0,
        team_0_id=10,
        team_1_id=20,
        team_0_players=[1, 2, 3],
        team_1_players=[4, 5, 6],
        frames=[
            Frame(
                ball=Coordinate(x=1.1, y=2.2, z=3.3),
                players={1: Coordinate(x=4.4, y=5.5, z=6.6), 2: Coordinate(x=7.7, y=8.8, z=9.9)},
            )
        ],
    )
    assert event == expected_event
