# Helper to download data for Alice example
from pathlib import Path

import pooch

from .._types import PathArg


def get_alice_path(
        path: PathArg = Path("~/Data/Alice"),
):
    path = Path(path).expanduser().resolve()
    path.mkdir(exist_ok=True, parents=True)

    baseurl = 'https://drum.lib.umd.edu/bitstream/handle/1903/27591/'
    registry = {
        'stimuli.zip': '92317dbfc81d6aef14fc334abd75d1165cf57501f0c11f8db1a47c76c3d90ac6',
        'eeg.1.zip': 'a645e4bf30ec8de10c92f82e9f842dd8172a4871f8eb23244e7e78b7dff157aa'
    }
    fetcher = pooch.Pooch(
        path=path,
        base_url=baseurl,
        registry=registry,
        retry_if_failed=4,
    )
    for fname in registry.keys():
        if (path / fname.split('.')[0]).exists():   # Won't work for multiple eeg.x.zip download
            continue
        fetcher.fetch(fname, processor=pooch.Unzip(extract_dir='.'))
        (path / fname).unlink()
    return path
