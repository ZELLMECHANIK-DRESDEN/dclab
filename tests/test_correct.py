"""Test correct-functions"""

from dclab import correct, new_dataset
import h5py
import numpy as np

from helper_methods import retrieve_data


def test_correct_offset():
    # Arrange
    path_in = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    path_out = path_in.with_name("corrected.rtdc")

    with h5py.File(path_in, 'r+') as hf:
        fl_max_old = hf["events"]["fl1_max"][:]
        fl_offset = min(np.min(hf["events"]["trace"]["fl1_raw"]), 1)

    # Act
    correct.offset(path_in=path_in, path_out=path_out)

    # Assert
    with new_dataset(path_out) as ds:
        assert "baseline 1 offset" in ds.config["fluorescence"]
        assert ds.config["fluorescence"]["baseline 1 offset"] == fl_offset
        assert (ds["fl1_max"] == fl_max_old - (fl_offset-1)).all()
