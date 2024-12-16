import h5py
import numpy as np
import pytest

import dclab
from dclab import new_dataset
from dclab.features.bright import get_bright

from helper_methods import calltracker, retrieve_data


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_brightness():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
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


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_brightness_called_once(monkeypatch):
    """Make sure dclab.features.bright.get_bright is only called once"""
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    # remove brightness features
    with h5py.File(path, "r+") as h5:
        del h5["events"]["bright_avg"]
        del h5["events"]["bright_sd"]

    # wrap the original brightness retrieval function
    old_get_bright = dclab.features.bright.get_bright

    @calltracker
    def wrap_get_bright(*args, **kwargs):
        return old_get_bright(*args, **kwargs)
    # Monkeypatch the imported function used in the feat_anc_core submodule,
    # otherwise (dclab.features.bright.get_bright) it is not called.
    monkeypatch.setattr("dclab.rtdc_dataset.feat_anc_core.af_image_contour"
                        ".features.bright.get_bright",
                        wrap_get_bright)

    # assert (access both brightness features and make sure that the
    # original function is only called once)
    ds = dclab.new_dataset(path)
    _ = ds["bright_sd"]
    _ = ds["bright_avg"]

    assert wrap_get_bright.calls == 1


def test_simple_bright():
    pytest.importorskip("nptdms")
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image-bright_2017.zip"))
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


def test_list_input_bright():
    """Use the list of np.ndarray as input."""
    pytest.importorskip("nptdms")
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image-bright_2017.zip"))
    # This stripped dataset has only 7 video frames / contours
    image, mask, bright_avg, bright_sd = [], [], [], []
    for ii in range(2, 7):
        image.append(ds["image"][ii])
        mask.append(ds["mask"][ii])
        bright_avg.append(ds["bright_avg"][ii])
        bright_sd.append(ds["bright_sd"][ii])

    avg, std = get_bright(mask=mask, image=image, ret_data="avg,sd")
    assert np.allclose(avg, bright_avg)
    assert np.allclose(std, bright_sd)
    # cover single `ret_data` input
    assert np.allclose(
        avg, get_bright(mask=mask, image=image, ret_data="avg"))
    assert np.allclose(
        std, get_bright(mask=mask, image=image, ret_data="sd"))
