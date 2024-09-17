import torch
import os
import pandas as pd
import pyarrow.parquet as pq
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional

from nba_tracking_data_commons.dataset.patched.data_config import PatchedDataConfig
from nba_tracking_data_commons.dataset.patched.utils import patchify
from nba_tracking_data_commons.dataset.transforms import shuffle_players, normalize_coordinates, flip_normalized_x_axis
from nba_tracking_data_commons.dataset.utils import get_data_split
from nba_tracking_data_commons.utils import list_files_in_directory


class PatchedParquetDataset(Dataset):
    def __init__(self, config: PatchedDataConfig, game_ids: List[str], stage: str = "train") -> None:
        self.config = config
        self.game_ids = game_ids
        self.stage = stage

        # Initialize data storage
        self.file_mapping = {}

        # Collect parquet file paths
        parquet_files = [
            os.path.join(config.input_path, f)
            for f in os.listdir(config.input_path)
            if f.endswith(".parquet") and f[:-8] in game_ids
        ]

        # Map parquet files
        for file_path in parquet_files:
            # Load parquet file
            table = pq.read_table(file_path)
            df = table.to_pandas()

            # Map data by (game_id, event_id)
            for (game_id, event_id), group in df.groupby(["game_id", "event_id"]):
                if (game_id, event_id) not in self.file_mapping:
                    self.file_mapping[(game_id, event_id)] = []
                self.file_mapping[(game_id, event_id)].append(file_path)

        # Sorting by (game_id, event_id) to ensure the access pattern
        self.event_ids = sorted(self.file_mapping.keys())

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        # Get game_id and event_id for this index
        game_id, event_id = self.event_ids[idx]

        # Retrieve file paths for this (game_id, event_id)
        file_paths = self.file_mapping[(game_id, event_id)]

        # Load data from parquet files
        dfs = []
        for file_path in file_paths:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            dfs.append(df[(df["game_id"] == game_id) & (df["event_id"] == event_id)])

        # Concatenate all dataframes
        df = pd.concat(dfs, ignore_index=True)

        # Extract flattened data
        flattened_data = df.drop(["game_id", "event_id"], axis=1).values

        # Determine number of time steps T
        T = len(flattened_data)

        # Reshape to (T, P, C)
        tracking_data = flattened_data.reshape(T, 11, 3)

        # Convert to PyTorch tensor
        x = torch.tensor(tracking_data, dtype=torch.float32)
        x = normalize_coordinates(x=x)

        if self.stage != "eval":
            x = shuffle_players(x=x, shuffle_players=self.config.shuffle_players)
            x = flip_normalized_x_axis(x=x)
        x = x.permute(2, 1, 0)  # (T, P, C) -> (C, P, T)
        x = patchify(x=x, patch_length=self.config.patch_size, padded_value=self.config.patch_pad_value)

        return {"game_id": game_id, "event_id": event_id, "tracking_data": x}


class PatchedCollateFn:
    def __init__(self, config: PatchedDataConfig, stage: str = "train"):
        self.event_length = config.event_length
        self.patch_pad_value = config.patch_pad_value
        self.stage = stage

    def __call__(self, batch):
        game_ids = [item["game_id"] for item in batch]
        event_ids = [item["event_id"] for item in batch]

        max_length = max([item["tracking_data"].shape[0] for item in batch])
        tracking_data = []
        for item in batch:
            x = item["tracking_data"]
            T, E = x.shape[0]  # (N, E)
            if T < max_length:
                # Pad with zeros to target_length
                padding_size = max_length - T
                padding = torch.full((padding_size, E), self.patch_pad_value)
                tracking_data += [torch.cat((x, padding), dim=0)]
            else:
                tracking_data += [x]
        tracking_data = torch.stack(tracking_data)

        return {"game_id": game_ids, "event_id": event_ids, "tracking_data": tracking_data}


class PatchedParquetDataModule(LightningDataModule):
    def __init__(self, config: PatchedDataConfig, stage: str = None) -> None:
        super().__init__()
        self.config = config

        self.train_dataset = None
        self.val_dataset = None
        self.setup(stage=stage)

    def setup(self, stage: Optional[str] = None):
        game_ids = list_files_in_directory(path=self.config.input_path, suffix=".parquet")

        data_split = get_data_split(
            config=self.config,
            game_ids=game_ids,
            stage=stage,
        )

        self.train_dataset = PatchedParquetDataset(
            config=self.config,
            game_ids=data_split["train"],
        )
        self.val_dataset = PatchedParquetDataset(
            config=self.config,
            game_ids=data_split["val"],
            stage="eval",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=PatchedCollateFn(config=self.config),
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            # shuffle=True,  # can't shuffle if using efficient memory
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            collate_fn=PatchedCollateFn(config=self.config),
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
        )


def setup_data_module(config: PatchedDataConfig, stage: str = None):
    return PatchedParquetDataModule(config=config, stage=stage)
