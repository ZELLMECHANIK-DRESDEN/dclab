import h5py
import numpy as np
import pytest

import dclab

from helper_methods import retrieve_data


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_brightness_bc():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    ds = dclab.new_dataset(path)
    # sanity checks
    assert "bright_perc_10" not in ds.features_innate
    assert "bright_perc_90" not in ds.features_innate
    # ignore first event (no image data)
    for ii in range(1, len(ds)):
        image_corr = np.array(ds["image"][ii], dtype=int) - ds["image_bg"][ii]
        mask = ds["mask"][ii]
        p10 = np.percentile(image_corr[mask], 10)
        p90 = np.percentile(image_corr[mask], 90)
        assert np.max(np.abs(image_corr)) < 50, "sanity check"
        assert ds["bright_perc_10"][ii] == p10
        assert ds["bright_perc_90"][ii] == p90


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_brightness_bc_with_offset():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    with h5py.File(path, "a") as h5:
        bg_off = np.linspace(0, 1, len(h5["events/deform"]))
        h5["events/bg_off"] = bg_off

    ds = dclab.new_dataset(path)
    # sanity checks
    assert "bright_perc_10" not in ds.features_innate
    assert "bright_perc_90" not in ds.features_innate
    # ignore first event (no image data)
    for ii in range(1, len(ds)):
        image_bg = ds["image_bg"][ii] + bg_off[ii]
        image_corr = np.array(ds["image"][ii], dtype=int) - image_bg
        mask = ds["mask"][ii]
        p10 = np.percentile(image_corr[mask], 10)
        p90 = np.percentile(image_corr[mask], 90)
        assert np.max(np.abs(image_corr)) < 50, "sanity check"
        assert ds["bright_perc_10"][ii] == p10
        assert ds["bright_perc_90"][ii] == p90


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_brightness_bc_single():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    ds = dclab.new_dataset(path)
    # sanity checks
    assert "bright_perc_10" not in ds.features_innate
    assert "bright_perc_90" not in ds.features_innate
    # ignore first event (no image data)
    for ii in range(1, len(ds)):
        image_corr = np.array(ds["image"][ii], dtype=int) - ds["image_bg"][ii]
        mask = ds["mask"][ii]
        p10 = np.percentile(image_corr[mask], 10)
        p90 = np.percentile(image_corr[mask], 90)
        assert np.max(np.abs(image_corr)) < 50, "sanity check"

        p10_calc, p90_calc = dclab.features.bright_perc.get_bright_perc(
            ds["mask"][ii],
            ds["image"][ii],
            ds["image_bg"][ii],
        )
        assert (p10_calc, p90_calc) == (p10, p90)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_brightness_bc_list():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    ds = dclab.new_dataset(path)
    # sanity checks
    assert "bright_perc_10" not in ds.features_innate
    assert "bright_perc_90" not in ds.features_innate
    # ignore first event (no image data)
    mask = list(ds["mask"][1:6])
    image = list(ds["image"][1:6])
    image_bg = list(ds["image_bg"][1:6])

    p10_calc, p90_calc = dclab.features.bright_perc.get_bright_perc(
        mask, image, image_bg,
    )
    assert isinstance(p10_calc, np.ndarray)
    assert isinstance(p90_calc, np.ndarray)
    assert p10_calc.shape == (4,)
    assert p90_calc.shape == (4,)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_af_brightness_bc_single_with_offset():
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    with h5py.File(path, "a") as h5:
        bg_off = np.linspace(0, 1, len(h5["events/deform"]))
        h5["events/bg_off"] = bg_off

    ds = dclab.new_dataset(path)
    # sanity checks
    assert "bright_perc_10" not in ds.features_innate
    assert "bright_perc_90" not in ds.features_innate
    # ignore first event (no image data)
    for ii in range(1, len(ds)):
        image_bg = ds["image_bg"][ii] + bg_off[ii]
        image_corr = np.array(ds["image"][ii], dtype=int) - image_bg
        mask = ds["mask"][ii]
        p10 = np.percentile(image_corr[mask], 10)
        p90 = np.percentile(image_corr[mask], 90)
        assert np.max(np.abs(image_corr)) < 50, "sanity check"
        assert dclab.features.bright_perc.get_bright_perc(
            ds["mask"][ii],
            ds["image"][ii],
            ds["image_bg"][ii],
            ds["bg_off"][ii],
        ) == (p10, p90)
