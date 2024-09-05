from dataclasses import dataclass


@dataclass
class Coordinate:
    x: float
    y: float
    z: float

    def to_row(self):
        return [self.x, self.y, self.z]

    @classmethod
    def from_dict(cls, data):
        return cls(x=data["x"], y=data["y"], z=data["z"])


@dataclass
class Frame:
    ball: Coordinate
    players: dict[int:Coordinate]

    @classmethod
    def from_dict(cls, data):
        return cls(
            ball=Coordinate.from_dict(data["ball"]),
            players={int(player_id): Coordinate.from_dict(coord) for player_id, coord in data["players"].items()},
        )


@dataclass
class Event:
    game_id: str
    event_id: int
    period: int
    game_clock_start: float
    game_clock_end: float
    wall_clock_start: float
    wall_clock_end: float
    team_0_id: int
    team_1_id: int
    team_0_players: list[int]
    team_1_players: list[int]
    frames: list[Frame]

    @classmethod
    def from_dict(cls, data):
        return cls(
            game_id=data["game_id"],
            event_id=data["event_id"],
            period=data["period"],
            game_clock_start=data["game_clock_start"],
            game_clock_end=data["game_clock_end"],
            wall_clock_start=data["wall_clock_start"],
            wall_clock_end=data["wall_clock_end"],
            team_0_id=data["team_0_id"],
            team_1_id=data["team_1_id"],
            team_0_players=data["team_0_players"],
            team_1_players=data["team_1_players"],
            frames=[Frame.from_dict(frame) for frame in data["frames"]],
        )
