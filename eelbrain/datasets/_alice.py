# Helper to download data for Alice example
import os
import pooch

from pathlib import Path
import zipfile

from .._types import PathArg


def get_alice_path(
        path: PathArg = Path("~/Data/Alice"),
):
    path = Path(path).expanduser().resolve()
    if path.exists():
        return path
    path.mkdir(exist_ok=True, parents=True)
    urls = [
        ['https://drum.lib.umd.edu/bitstream/handle/1903/27591/stimuli.zip', '92317dbfc81d6aef14fc334abd75d1165cf57501f0c11f8db1a47c76c3d90ac6'],
        # ['https://drum.lib.umd.edu/bitstream/handle/1903/27591/eeg.0.zip', 'd63d96a6e5080578dbf71320ddbec0a0'],
        ['https://drum.lib.umd.edu/bitstream/handle/1903/27591/eeg.1.zip', 'a645e4bf30ec8de10c92f82e9f842dd8172a4871f8eb23244e7e78b7dff157aa'],  # S15-S34
        # ['https://drum.lib.umd.edu/bitstream/handle/1903/27591/eeg.2.zip', '3fb33ca1c4640c863a71bddd45006815'],
    ]

    for url, known_hash in urls:
        temp_file_path = pooch.retrieve(url, known_hash)
        with zipfile.ZipFile(temp_file_path, 'r') as f:
            f.extractall(path)
        os.remove(temp_file_path)
    return path
