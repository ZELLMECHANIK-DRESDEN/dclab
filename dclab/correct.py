"""Functions used to correct RTDC-files"""

import pathlib
import shutil

import h5py
import numpy as np


def offset(path_in, path_out):
    """Computes trace offset and corrects features "fl?_max" accordingly

    Parameters
    ----------
    path_in: str or pathlib.Path
        Path of input file for which the offset will be corrected.
    path_out: str or pathlib.Path
        Path where file with corrected offset will be saved
    """
    path_in = pathlib.Path(path_in)
    path_out = pathlib.Path(path_out)

    if path_out.suffix != ".rtdc":
        path_out = path_out.with_name(path_out.name + ".rtdc")

    if path_out.exists():
        raise ValueError("Output file '{}' already exists!".format(path_out))

    shutil.copy2(path_in, path_out)

    with h5py.File(path_out, 'r+') as hf:
        ds = hf["events"]

        for ii in range(1, 4):
            # skip if offset already corrected
            feat = "fl{}_raw".format(ii)
            feat_max = "fl{}_max".format(ii)
            baseline_str = "fluorescence:baseline {} offset".format(ii)

            if baseline_str in hf.attrs:
                continue

            if feat in ds["trace"] and feat_max in ds:
                trace = ds["trace"][feat]
                offset = np.min(trace)
                # Only consider negative offsets
                offset = min(offset, 1)
                hf.attrs["fluorescence:baseline {} offset".format(ii)] = offset
                ds["fl{}_max".format(ii)][:] -= (offset-1)
