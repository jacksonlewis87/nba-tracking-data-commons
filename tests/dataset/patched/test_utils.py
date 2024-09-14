import pytest
import torch

from nba_tracking_data_commons.dataset.patched.utils import patchify, unpatchify


@pytest.mark.parametrize(
    "padded_value, expected_result",
    [
        (
            None,
            torch.tensor(
                [
                    [0, 10, 20, 1, 11, 21],
                    [4, 14, 24, 5, 15, 25],
                    [2, 12, 22, 3, 13, 23],
                    [6, 16, 26, 7, 17, 27],
                ]
            ),
        ),
        (
            -1,
            torch.tensor(
                [
                    [0, 10, 20, 1, 11, 21],
                    [4, 14, 24, 5, 15, 25],
                    [2, 12, 22, 3, 13, 23],
                    [6, 16, 26, 7, 17, 27],
                    [-3, -13, -23, -1, -1, -1],
                    [-7, -17, -27, -1, -1, -1],
                ]
            ),
        ),
    ],
)
def test_patchify(padded_value, expected_result):
    # (C, P, F)
    dummy_tracking_data = torch.tensor(
        [
            [
                [0, 1, 2, 3, -3],  # p0
                [4, 5, 6, 7, -7],  # p1
            ],  # c0
            [
                [10, 11, 12, 13, -13],  # p0
                [14, 15, 16, 17, -17],  # p1
            ],  # c1
            [
                [20, 21, 22, 23, -23],  # p0
                [24, 25, 26, 27, -27],  # p1
            ],  # c2
        ]
    )

    result = patchify(
        x=dummy_tracking_data,
        patch_length=2,
        padded_value=padded_value,
    )

    assert torch.all(torch.eq(result, expected_result))


def test_unpatchify():
    # (C, P, F)
    dummy_tracking_data = torch.tensor(
        [
            [
                [0, 1, 2, 3],  # p0
                [4, 5, 6, 7],  # p1
            ],  # c0
            [
                [10, 11, 12, 13],  # p0
                [14, 15, 16, 17],  # p1
            ],  # c1
            [
                [20, 21, 22, 23],  # p0
                [24, 25, 26, 27],  # p1
            ],  # c2
        ]
    )
    x_patched = torch.tensor(
        [
            [0, 10, 20, 1, 11, 21],
            [4, 14, 24, 5, 15, 25],
            [2, 12, 22, 3, 13, 23],
            [6, 16, 26, 7, 17, 27],
        ]
    )

    result = unpatchify(
        x=x_patched,
        channels=3,
        num_players=2,
        patch_length=2,
    )

    assert torch.all(torch.eq(result, dummy_tracking_data))
