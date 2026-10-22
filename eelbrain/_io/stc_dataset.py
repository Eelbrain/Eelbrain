import os
import re
import glob
import itertools

from mne import read_source_estimate

from .._data_obj import Dataset, Factor
from .fiff import stc_ndvar


class DatasetSTCLoader:
    """
    Load source estimates on disk into Dataset for use in statistical tests

    Parameters
    ----------
    data_dir
        Path to directory containing stc files

    Attributes
    ----------
    data_dir
        Path to data directory
    subjects
        Subject IDs extracted from stc filenames
    factors
        Names of experimental factors
    levels
        Names of levels of each factor in ``factors``

    Notes
    -----
    When instantiated, the loader will automatically do level detection
    based on .stc filenames. The user must explicitly set the factor
    names with :meth:`DatasetSTCLoader.set_factor_names`. The dataset
    may then be loaded via :meth:`DatasetSTCLoader.make_dataset`.

    Examples
    --------
    >>> loader = DatasetSTCLoader("path/to/exported/stcs")
    >>> loader.set_factor_names(["factor1", "factor2"])
    >>> data = loader.make_dataset(subjects_dir="mri/")

    See Also
    --------
    eelbrain.gui.load_stcs : a GUI to load source estimates into a Dataset
    """
    data_dir: str
    subjects: tuple[str, ...]
    factors: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]

    def __init__(self, data_dir: str):
        if not os.path.exists(data_dir):
            raise ValueError(f"Directory '{data_dir}' not found.")
        self.data_dir = data_dir
        self.subjects = None
        self.levels = None
        self.factors = None
        self._n_factors = None
        self._level_lens = None
        self._find_subjects()
        self._find_level_names()

    def __repr__(self):
        tmp = "<DatasetSTCLoader: {} subjects | {} design>"
        return tmp.format(len(self.subjects), self.design_shape)

    def _all_stc_filenames(self):
        return glob.glob(os.path.join(self.data_dir, "*", "*.stc"))

    def _find_subjects(self):
        pattern = re.compile(r"[AR]\d{4}")
        stcs = self._all_stc_filenames()
        subjects = {pattern.search(s).group() for s in stcs}
        self.subjects = tuple(subjects)

    def _find_level_names(self):
        stcs = self._all_stc_filenames()
        if not stcs:
            raise ValueError("No .stc files in sub-directories")
        # condition names should be lowest level folder
        cond_dirs = list({s.split(os.sep)[-2] for s in stcs})
        # set number of factors based on first full condition name
        self._n_factors = len(cond_dirs[0].split("_"))
        splits = (c.split("_") for c in cond_dirs)
        # transpose to group level names by factor; keep unique
        cond_sets = list(map(set, zip(*splits)))
        self.levels = tuple(tuple(c) for c in cond_sets)  # list of tuples, not sets
        self._level_lens = [len(lev) for lev in self.levels]

    def set_factor_names(self, factors: list[str] | tuple[str, ...]):
        """
        Set names of experimental factors

        Parameters
        ----------
        factors
            Factor names. Length must match the number of factors detected
            from stc filenames.
        """
        if not self.levels:
            raise RuntimeError("No level names were detected from "
                               "the files in the data directory.")
        if len(factors) != self._n_factors:
            msg = ("There were %d factors detected, but %d factor "
                   "names provided." % (self._n_factors, len(factors)))
            raise ValueError(msg)
        self.factors = tuple(factors)

    @property
    def design_shape(self):
        """Shape of experiment design, e.g. '2 x 3'"""
        if self.levels is None:
            return None
        des = " x ".join(map(str, self._level_lens))
        if len(des) == 1:
            des = f"1 x {des}"
        return des

    def make_dataset(self, load_stcs: bool = True, subject: str = "fsaverage",
                     src: str = "ico-4", **stc_kwargs):
        """
        Load stcs into a Dataset with columns for subject and experimental factors

        Dataset contains one case per condition per subject, and source estimates
        loaded as an NDVar. Any additional keyword arguments are passed to
        :meth:`eelbrain.load.mne.stc_ndvar`. If ``SUBJECTS_DIR`` is not set in your
        environment, it should be provided here.

        Parameters
        ----------
        load_stcs
            Whether to include stc data in dataset. Only False when testing
            on unreadable stc files.
        subject
            Subject ID of brain to which the source estimates belong;
            default: 'fsaverage'
        src
            Source space surface decimation; default 'ico-4'

        Returns
        -------
        data : eelbrain.Dataset
            Dataset with columns 'subject' (random factor), 'src' (NDVar of stc data),
            and one Factor for each item in ``self.factors``.
        """
        rows = itertools.product(self.subjects, *self.levels)
        columns = map(Factor, zip(*rows))
        col_names = ["subject"] + list(self.factors)
        data = Dataset(zip(col_names, columns))
        data["subject"].random = True
        stc_fnames = []
        for c in data.itercases():
            folder = "_".join(c[i] for i in self.factors)
            exp = f"{self.data_dir}/{folder}/{c['subject']}*-lh.stc"
            fnames = glob.glob(exp)
            assert len(fnames) == 1
            stc_fnames.append(fnames[0])
        if load_stcs:
            stcs = list(map(read_source_estimate, stc_fnames))
            data["src"] = stc_ndvar(stcs, subject=subject, src=src, **stc_kwargs)
        return data
