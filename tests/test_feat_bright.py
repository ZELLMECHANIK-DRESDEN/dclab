import h5py
import numpy as np

import dclab
from dclab import new_dataset
from dclab.features.bright import get_bright

from helper_methods import retrieve_data


def test_af_brightness():
    path = retrieve_data("rtdc_data_hdf5_image_bg.zip")
    with h5py.File(path, "r+") as h5:
        real_avg = h5["events"]["bright_avg"][:]
        real_sd = h5["events"]["bright_sd"][:]
        del h5["events"]["bright_avg"]
        del h5["events"]["bright_sd"]
    ds = dclab.new_dataset(path)
    # sanity checks
    assert "bright_avg" not in ds.features_innate
    assert "bright_sd" not in ds.features_innate
    comp_avg = ds["bright_avg"]
    comp_sd = ds["bright_sd"]
    idcompare = ~np.isnan(comp_avg)
    # ignore first event (no image data)
    idcompare[0] = False
    assert np.allclose(real_avg[idcompare], comp_avg[idcompare])
    assert np.allclose(real_sd[idcompare], comp_sd[idcompare])


def test_simple_bright():
    ds = new_dataset(retrieve_data("rtdc_data_traces_video_bright.zip"))
    for ii in range(2, 7):
        # This stripped dataset has only 7 video frames / contours
        image = ds["image"][ii]
        mask = ds["mask"][ii]
        avg, std = get_bright(mask=mask, image=image, ret_data="avg,sd")
        assert np.allclose(avg, ds["bright_avg"][ii])
        assert np.allclose(std, ds["bright_sd"][ii])
        # cover single `ret_data` input
        assert np.allclose(
            avg, get_bright(mask=mask, image=image, ret_data="avg"))
        assert np.allclose(
            std, get_bright(mask=mask, image=image, ret_data="sd"))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
