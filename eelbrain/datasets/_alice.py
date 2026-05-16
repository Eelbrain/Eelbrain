# Helper to download data for Alice example
from pathlib import Path
import sys

import pooch
from tqdm import tqdm

from .._types import PathArg


def get_alice_path(
        path: PathArg = Path("~/Data/Alice"),
        *,
        progressbar: bool = False,
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
    for fname in registry:
        if (path / fname.split('.')[0]).exists():   # Won't work for multiple eeg.x.zip download
            continue
        # Avoid 403 errors from the server by setting a user agent
        # adapted from https://github.com/scipy/scipy/pull/22076
        progress = SlowTqdm() if progressbar else None
        downloader = pooch.HTTPDownloader(progressbar=progress, headers={"User-Agent": "Eelbrain"})
        fetcher.fetch(fname, processor=pooch.Unzip(extract_dir='.'), downloader=downloader)
        (path / fname).unlink()
    return path


# Adapt base progressbar code:
# https://github.com/fatiando/pooch/blob/main/pooch/downloaders.py#L238
# To update more slowly
class SlowTqdm(tqdm):
    def __init__(self):
        use_ascii = bool(sys.platform == "win32")
        super().__init__(
            total=1,  # just to make bool() happy
            mininterval=1.0,  # slower than default 0.1
            ascii=use_ascii,
            unit="B",
            unit_scale=True,
            leave=True,
        )
