import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional

from nba_tracking_data_commons.dataset.patched.data_config import PatchedDataConfig
from nba_tracking_data_commons.dataset.patched.utils import patchify
from nba_tracking_data_commons.dataset.transforms import shuffle_players, normalize_coordinates, flip_normalized_x_axis
from nba_tracking_data_commons.dataset.utils import get_data_split
from nba_tracking_data_commons.utils import list_files_in_directory, load_tensor


class PatchedDataset(Dataset):
    def __init__(self, config: PatchedDataConfig, game_ids: list[str], stage: str = "train") -> None:
        self.config = config
        self.game_ids = game_ids
        self.stage = stage

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        game_events, event_sizes = load_tensor(path=self.config.input_path, tensor_name=self.game_ids[index])
        if not self.config.include_z:
            game_events = game_events[:, :, :2]

        game_events = normalize_coordinates(x=game_events)

        x = []
        for event_size in event_sizes:
            event = game_events[:event_size]
            game_events = game_events[event_size:]
            if self.stage != "eval":
                event = shuffle_players(x=event, shuffle_players=self.config.shuffle_players)
                event = flip_normalized_x_axis(x=event)
            event = patchify(x=event, patch_length=self.config.patch_size, padded_value=self.config.patch_pad_value)
            x += [event]

        return x

    def __len__(self) -> int:
        return len(self.game_ids)


class PatchedDataModule(LightningDataModule):
    def __init__(self, config: PatchedDataConfig, stage: str = None) -> None:
        super().__init__()
        self.config = config

        self.train_dataset = None
        self.val_dataset = None
        self.setup(stage=stage)

    def setup(self, stage: Optional[str] = None):
        game_ids = list_files_in_directory(path=self.config.input_path, suffix=".pt")

        data_split = get_data_split(
            config=self.config,
            game_ids=game_ids,
            stage=stage,
        )

        self.train_dataset = PatchedDataset(
            config=self.config,
            game_ids=data_split["train"],
        )
        self.val_dataset = PatchedDataset(
            config=self.config,
            game_ids=data_split["val"],
            stage="eval",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size)


def setup_data_module(config: PatchedDataConfig, stage: str = None):
    return PatchedDataModule(config=config, stage=stage)
