import math
import torch
import torch.nn.functional as F


def patchify(x: torch.Tensor, patch_length: int, padded_value: float = None):
    # Input tensor dimensions
    channels, num_players, num_frames = x.shape

    # Calculate number of sequence patches
    num_sequence_patches = (
        math.ceil(num_frames / patch_length) if padded_value else math.floor(num_frames / patch_length)
    )

    # Calculate the necessary size for padding or cropping
    new_num_frames = num_sequence_patches * patch_length

    if padded_value is not None:
        # Pad the tensor to ensure full patches
        padding = (0, new_num_frames - num_frames)
        x = F.pad(x, pad=padding, mode="constant", value=padded_value)
    else:
        # Crop the tensor to ensure full patches
        x = x[:, :, :new_num_frames]

    # Reshape the tensor
    x = x.reshape(shape=(channels, num_players, num_sequence_patches, patch_length))

    # Permute dimensions to (num_sequence_patches * num_players, patch_length * channels)
    x = torch.einsum("cpsl->splc", x)
    x = x.reshape((num_sequence_patches * num_players, patch_length * channels))

    return x


def unpatchify(
    x: torch.Tensor,
    channels: int,
    num_players: int,
    patch_length: int,
):
    num_sequence_patches = x.shape[1] // channels

    # Calculate the original number of frames based on the number of patches and patch length
    original_num_frames = num_sequence_patches * patch_length

    # Reverse the final reshape and permutation
    x = x.reshape((num_sequence_patches, num_players, patch_length, channels))
    x = torch.einsum("splc->cpsl", x)

    # Reshape back to (channels, num_players, num_frames)
    x = x.reshape((channels, num_players, original_num_frames))

    return x
