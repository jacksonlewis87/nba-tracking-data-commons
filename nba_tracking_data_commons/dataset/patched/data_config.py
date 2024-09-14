from dataclasses import dataclass
from typing import Optional

from pytorch_model_commons.data.data_config import DataConfig


@dataclass
class PatchedDataConfig(DataConfig):
    patch_size: int
    shuffle_players: bool
    include_z: bool
    patch_pad_value: Optional[float]
