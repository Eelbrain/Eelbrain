# Helper to download data for Alice example
from pathlib import Path

import pooch

from .._types import PathArg


def get_alice_path(
        path: PathArg = Path("~/Data/Alice"),
):
    path = Path(path).expanduser().resolve()
    path.mkdir(exist_ok=True, parents=True)

    registry = {
        'stimuli.zip': '92317dbfc81d6aef14fc334abd75d1165cf57501f0c11f8db1a47c76c3d90ac6',
        'eeg.1.zip': 'a645e4bf30ec8de10c92f82e9f842dd8172a4871f8eb23244e7e78b7dff157aa'
    }
    urls = {
        'stimuli.zip': 'https://drum.lib.umd.edu/bitstreams/df241468-26ee-42df-b27f-3f438cfc5a3f/download',
        'eeg.1.zip': 'https://drum.lib.umd.edu/bitstreams/bef532d8-cf74-4b9d-9b4c-5c1f81610ce9/download',
    }
    fetcher = pooch.Pooch(
        path=path,
        base_url='',
        urls=urls,
        registry=registry,
        retry_if_failed=4,
    )
    downloader = pooch.HTTPDownloader(headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'})
    for fname in registry.keys():
        if (path / fname.split('.')[0]).exists():   # Won't work for multiple eeg.x.zip download
            continue
        fetcher.fetch(fname, processor=pooch.Unzip(extract_dir='.'), downloader=downloader)
        (path / fname).unlink()
    return path
