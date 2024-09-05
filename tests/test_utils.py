from dataclasses import dataclass
import json
import pytest
from unittest.mock import Mock, patch, mock_open

from utils import DataClassEncoder, load_json, write_json, list_files_in_directory, load_tensor


def test_dataclass_encoder():
    @dataclass
    class DummyDataClass0:
        field_0: int
        field_1: str

    @dataclass
    class DummyDataClass1:
        field_0: int
        field_1: DummyDataClass0

    data = DummyDataClass1(
        field_0=1,
        field_1=DummyDataClass0(
            field_0=2,
            field_1="some-value",
        ),
    )

    result = json.dumps(data, cls=DataClassEncoder)

    assert result == '{"field_0": 1, "field_1": {"field_0": 2, "field_1": "some-value"}}'


@patch("builtins.open", new_callable=mock_open)
@patch("utils.json")
def test_load_json(mock_json, mock_builtins_open):
    path = "dummy_path"

    result = load_json(path=path)

    mock_builtins_open.assert_called_once_with(path, "r")
    mock_json.load.assert_called_once_with(fp=mock_builtins_open.return_value)
    assert result == mock_json.load.return_value


@pytest.mark.parametrize("encoder", [None, Mock()])
@patch("builtins.open", new_callable=mock_open)
@patch("utils.json")
def test_write_json(mock_json, mock_builtins_open, encoder):
    path = "dummy_path"
    json_object = Mock()

    if encoder:
        write_json(path=path, json_object=json_object, encoder=encoder)
    else:
        write_json(path=path, json_object=json_object)

    mock_builtins_open.assert_called_once_with(path, "w")
    mock_json.dump.assert_called_once_with(json_object, cls=encoder, fp=mock_builtins_open.return_value)


@pytest.mark.parametrize(
    "suffix, expected_result",
    [
        (None, ["file1.txt", "file2.csv", "file3.txt"]),
        (".txt", ["file1", "file3"]),
    ],
)
@patch("utils.os.listdir", return_value=["file1.txt", "file2.csv", "file3.txt"])
def test_list_files_in_directory(mock_listdir, suffix, expected_result):
    path = "dummy_path"

    result = list_files_in_directory(path=path, suffix=suffix)

    mock_listdir.assert_called_once_with(path)
    assert result == expected_result


@patch("utils.torch_load")
def test_load_tensor(mock_torch_load):
    path = "dummy_path"
    tensor_name = "some-tensor"

    result = load_tensor(path=path, tensor_name=tensor_name)

    mock_torch_load.assert_called_once_with("dummy_path/some-tensor.pt")
    assert result == mock_torch_load.return_value
