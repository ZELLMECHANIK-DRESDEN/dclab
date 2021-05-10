import pathlib
import h5py
import numpy as np
import pytest

import dclab
from dclab.rtdc_dataset.plugins.plugin_feature import (
    PlugInFeature, import_plugin_feature_script,
    remove_plugin_feature, remove_all_plugin_features,
    PluginImportError)
from dclab.rtdc_dataset.ancillaries.ancillary_feature import (
    BadFeatureSizeWarning)

from helper_methods import retrieve_data

data_dir = pathlib.Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def cleanup_plugin_features():
    """Fixture used to setup and cleanup some fake ancillary features"""
    # code run before the test
    pass
    # then the test is run
    yield
    # code run after the test
    # remove our test plugin examples
    remove_all_plugin_features()


def compute_single_plugin_feature(rtdc_ds):
    circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
    return circ_per_area


def compute_multiple_plugin_features(rtdc_ds):
    circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
    circ_times_area = rtdc_ds["circ"] * rtdc_ds["area_um"]
    return {"circ_per_area": circ_per_area, "circ_times_area": circ_times_area}


def example_plugin_info_single_feature():
    info = {
        "method": compute_single_plugin_feature,
        "description": "This plugin will compute a feature",
        "long description": "Even longer description that "
                            "can span multiple lines",
        "feature names": ["circ_per_area"],
        "feature labels": ["Circularity per Area"],
        "features required": ["circ", "area_um"],
        "config required": [],
        "method check required": lambda x: True,
        "scalar feature": [True],
        "version": "0.1.0",
    }
    return info


def example_plugin_info_multiple_feature():
    info = {
        "method": compute_multiple_plugin_features,
        "description": "This plugin will compute some features",
        "long description": "Even longer description that "
                            "can span multiple lines",
        "feature names": ["circ_per_area", "circ_times_area"],
        "feature labels": ["Circularity per Area", "Circularity times Area"],
        "features required": ["circ", "area_um"],
        "config required": [],
        "method check required": lambda x: True,
        "scalar feature": [True, True],
        "version": "0.1.0",
    }
    return info


def test_pf_attributes():
    plugin_path = data_dir / "plugin_test_example.py"
    plugin_list = dclab.load_plugin_feature(plugin_path)
    pf1, pf2 = plugin_list

    plugin_file_info = import_plugin_feature_script(plugin_path)

    assert pf1.feature_name == plugin_file_info["feature names"][0]
    assert pf2.feature_name == plugin_file_info["feature names"][1]
    assert pf1._plugin_feature_name == plugin_file_info["feature names"][0]
    assert pf2._plugin_feature_name == plugin_file_info["feature names"][1]
    assert pf1.plugin_path == plugin_path
    assert pf2.plugin_path == plugin_path
    assert pf1._plugin_original_info == plugin_file_info
    assert pf2._plugin_original_info == plugin_file_info


def test_pf_attribute_ancill_info():
    info = example_plugin_info_single_feature()
    pf = PlugInFeature("circ_per_area", info)
    # comparing lambda functions fails due to differing memory locations
    pf._ancill_info.pop("req_func")
    ancill_info = {
        "feature_name": "circ_per_area",
        "method": compute_single_plugin_feature,
        "req_config": [],
        "req_features": ["circ", "area_um"],
    }
    assert pf._ancill_info == ancill_info


def test_pf_attribute_plugin_feature_info():
    info = example_plugin_info_single_feature()
    # comparing lambda functions fails due to differing memory locations
    info.pop("method check required")
    pf = PlugInFeature("circ_per_area", info)
    pf.plugin_feature_info.pop("method check required")
    plugin_feature_info = {
        "method": compute_single_plugin_feature,
        "description": "This plugin will compute a feature",
        "long description": "Even longer description that "
                            "can span multiple lines",
        "feature name": "circ_per_area",
        "feature label": "Circularity per Area",
        "features required": ["circ", "area_um"],
        "config required": [],
        "scalar feature": True,
        "version": "0.1.0",
    }
    assert pf.plugin_feature_info == plugin_feature_info


def test_pf_bad_plugin_feature_name():
    """Basic test of a bad name for PlugInFeature"""
    info = example_plugin_info_single_feature()
    info["feature names"] = "Peter-Pan's Best Friend!"
    with pytest.raises(ValueError):
        PlugInFeature("Peter-Pan's Best Friend!", info)


def test_pf_exists_in_hierarchy():
    """Test for RTDCHierarchy"""
    info = example_plugin_info_single_feature()
    pf = PlugInFeature("circ_per_area", info)
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with dclab.new_dataset(h5path) as ds:
        assert pf.feature_name in ds
        assert dclab.dfn.feature_exists(pf.feature_name)
        child = dclab.new_dataset(ds)
        assert pf.feature_name in child


def test_pf_export_and_load():
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    # initialize PlugInFeature instance
    info = example_plugin_info_single_feature()
    pf = PlugInFeature("circ_per_area", info)

    with dclab.new_dataset(h5path) as ds:
        # extract the feature information from the dataset
        assert pf in PlugInFeature.features
        circ_per_area = ds[pf.feature_name]

        # export the data to a new file
        expath = h5path.with_name("exported.rtdc")
        ds.export.hdf5(expath, features=ds.features_innate + [pf.feature_name])

    # make sure that worked
    with h5py.File(expath, "r") as h5:
        assert pf.feature_name in h5["events"]
        assert np.allclose(h5["events"][pf.feature_name], circ_per_area)

    # now check again with dclab
    with dclab.new_dataset(expath) as ds2:
        assert pf in PlugInFeature.features
        assert pf.feature_name in ds2
        assert pf.feature_name in ds2.features_innate
        assert np.allclose(ds2[pf.feature_name], circ_per_area)

        # and a control check
        remove_plugin_feature(pf)
        assert pf.feature_name not in ds2


def test_pf_feature_exists():
    plugin_path = data_dir / "plugin_test_example.py"
    plugin_list = dclab.load_plugin_feature(plugin_path)
    assert dclab.dfn.feature_exists(plugin_list[0].feature_name)
    assert dclab.dfn.feature_exists(plugin_list[1].feature_name)


def test_pf_filtering_with_plugin_feature():
    """Filtering with features"""
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with dclab.new_dataset(h5path) as ds:
        info = example_plugin_info_single_feature()
        pf = PlugInFeature("circ_per_area", info)

        ds.config["filtering"][f"{pf.feature_name} min"] = 0.030
        ds.config["filtering"][f"{pf.feature_name} max"] = 0.031
        ds.apply_filter()
        assert np.sum(ds.filter.all) == 1
        assert ds.filter.all[4]


def test_pf_import_plugin_info():
    plugin_path = data_dir / "plugin_test_example.py"
    info = import_plugin_feature_script(plugin_path)
    assert isinstance(info, dict)


def test_pf_import_plugin_info_bad_path():
    bad_plugin_path = "not/a/real/path/plugin.py"
    with pytest.raises(PluginImportError):
        _ = import_plugin_feature_script(bad_plugin_path)


def test_pf_incorrect_input_info():
    info = ["this", "is", "not", "a", "dict"]
    with pytest.raises(ValueError):
        PlugInFeature("feature_1", info)


def test_pf_incorrect_input_feature_name():
    info = example_plugin_info_single_feature()
    # `feature_name` is "circ_per_area" in info
    with pytest.raises(ValueError):
        PlugInFeature("not_the_correct_name", info)


def test_pf_incorrect_input_feature_labels():
    info = example_plugin_info_single_feature()
    info.pop("feature labels")
    with pytest.raises(ValueError):
        PlugInFeature("circ_per_area", info)


def test_pf_incorrect_input():
    info = example_plugin_info_single_feature()
    # set `info["method"]` to something that isn't callable
    info["method"] = "this_is_a_string"
    with pytest.raises(ValueError):
        PlugInFeature("circ_per_area", info)


def test_pf_initialize_plugin_after_loading():
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with dclab.new_dataset(h5path) as ds:
        circ_per_area = compute_single_plugin_feature(ds)
    with h5py.File(h5path, "a") as h5:
        h5["events"]["circ_per_area"] = circ_per_area
    with dclab.new_dataset(h5path) as ds:
        assert "circ_per_area" not in ds
        info = example_plugin_info_single_feature()
        PlugInFeature("circ_per_area", info)
        assert "circ_per_area" in ds
        assert "circ_per_area" in ds.features_innate


def test_pf_initialize_plugin_feature_single():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    info = example_plugin_info_single_feature()
    PlugInFeature("circ_per_area", info)
    assert "circ_per_area" in ds

    circ_per_area = ds["circ_per_area"]
    assert np.allclose(circ_per_area, ds["circ"] / ds["area_um"])

    # check that PlugInFeature exists independent of loaded ds
    with pytest.raises(AssertionError):
        ds2 = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
        assert "circ_per_area" not in ds2


def test_pf_initialize_plugin_features_multiple():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    info = example_plugin_info_multiple_feature()
    PlugInFeature("circ_per_area", info)
    PlugInFeature("circ_times_area", info)

    assert "circ_per_area" in ds
    assert "circ_per_area" not in ds.features_innate
    assert "circ_times_area" in ds
    assert "circ_times_area" not in ds.features_innate
    assert dclab.dfn.feature_exists("circ_per_area")
    assert dclab.dfn.feature_exists("circ_times_area")
    circ_per_area = ds["circ_per_area"]
    circ_times_area = ds["circ_times_area"]
    assert np.allclose(circ_per_area, ds["circ"] / ds["area_um"])
    assert np.allclose(circ_times_area, ds["circ"] * ds["area_um"])


def test_pf_load_plugin():
    plugin_path = data_dir / "plugin_test_example.py"
    plugin_list = dclab.load_plugin_feature(plugin_path)
    assert isinstance(plugin_list[0], PlugInFeature)
    assert isinstance(plugin_list[1], PlugInFeature)

    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert "circ_per_area" in ds
    assert "circ_per_area" not in ds.features_innate
    assert "circ_times_area" in ds
    assert "circ_times_area" not in ds.features_innate
    circ_per_area = ds["circ_per_area"]
    circ_times_area = ds["circ_times_area"]
    assert np.allclose(circ_per_area, ds["circ"] / ds["area_um"])
    assert np.allclose(circ_times_area, ds["circ"] * ds["area_um"])


def test_pf_load_plugin_bad_path():
    bad_plugin_path = "not/a/real/path/plugin.py"
    with pytest.raises(PluginImportError):
        _ = dclab.load_plugin_feature(bad_plugin_path)


def test_pf_remove_plugin_feature():
    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert "circ_per_area" not in ds
    assert "circ_times_area" not in ds

    plugin_path = data_dir / "plugin_test_example.py"
    plugin_list = dclab.load_plugin_feature(plugin_path)
    assert len(plugin_list) == 2
    assert "circ_per_area" in ds
    assert "circ_per_area" not in ds.features_innate
    assert "circ_times_area" in ds
    assert "circ_times_area" not in ds.features_innate
    assert dclab.dfn.feature_exists("circ_per_area")
    assert dclab.dfn.feature_exists("circ_times_area")

    remove_plugin_feature(plugin_list[0])
    remove_plugin_feature(plugin_list[1])

    assert "circ_per_area" not in ds
    assert "circ_times_area" not in ds
    assert not dclab.dfn.feature_exists("circ_per_area")
    assert not dclab.dfn.feature_exists("circ_times_area")

    with pytest.raises(TypeError):
        not_a_plugin_instance = [4, 6, 5]
        remove_plugin_feature(not_a_plugin_instance)


def test_pf_remove_all_plugin_features():
    plugin_path = data_dir / "plugin_test_example.py"
    _ = dclab.load_plugin_feature(plugin_path)
    ds = dclab.new_dataset(retrieve_data("rtdc_data_hdf5_rtfdc.zip"))
    assert "circ_per_area" in ds
    assert "circ_per_area" not in ds.features_innate
    assert "circ_times_area" in ds
    assert "circ_times_area" not in ds.features_innate
    assert dclab.dfn.feature_exists("circ_per_area")
    assert dclab.dfn.feature_exists("circ_times_area")

    remove_all_plugin_features()

    assert "circ_per_area" not in ds
    assert "circ_times_area" not in ds
    assert not dclab.dfn.feature_exists("circ_per_area")
    assert not dclab.dfn.feature_exists("circ_times_area")


def test_pf_try_existing_feature_fails():
    """Basic test of a temporary feature"""
    info = example_plugin_info_single_feature()
    info["feature names"] = ["deform"]
    with pytest.raises(ValueError):
        PlugInFeature("deform", info)


def test_pf_with_empty_feature_label_string():
    """Show that an empty `feature_label` will still give a descriptive
    feature label. See `dclab.dfn._add_feature_to_definitions` for details.
    """
    info = example_plugin_info_single_feature()
    info["feature labels"] = [""]
    feature_name = "circ_per_area"
    PlugInFeature(feature_name, info)
    label = dclab.dfn.get_feature_label("circ_per_area")
    assert dclab.dfn.feature_exists("circ_per_area")
    assert label != ""
    assert label == "User defined feature {}".format(feature_name)


def test_pf_with_no_feature_label():
    """Show that an empty `feature_label` will still give a descriptive
    feature label. See `dclab.dfn._add_feature_to_definitions` for details.
    """
    info = example_plugin_info_single_feature()
    info["feature labels"] = [None]
    feature_name = "circ_per_area"
    PlugInFeature(feature_name, info)
    label = dclab.dfn.get_feature_label("circ_per_area")
    assert dclab.dfn.feature_exists("circ_per_area")
    assert label is not None
    assert label == "User defined feature {}".format(feature_name)


def test_pf_wrong_data_shape_1():
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with dclab.new_dataset(h5path) as ds:
        info = example_plugin_info_single_feature()
        info["scalar feature"] = [False]
        pf = PlugInFeature("circ_per_area", info)
        with pytest.raises(ValueError):
            _ = ds[pf.feature_name]


def test_pf_wrong_data_shape_2():
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with dclab.new_dataset(h5path) as ds:
        info = example_plugin_info_single_feature()
        info["scalar feature"] = [True]
        info["method"] = lambda x: np.arange(len(ds)*2).reshape(-1, 2)
        pf = PlugInFeature("circ_per_area", info)
        with pytest.raises(ValueError):
            _ = ds[pf.feature_name]


def test_pf_wrong_length_1():
    """temporary feature should have same length"""
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with dclab.new_dataset(h5path) as ds:
        info = example_plugin_info_single_feature()
        info["method"] = lambda x: np.arange(len(ds)//2)
        pf = PlugInFeature("circ_per_area", info)
        with pytest.warns(BadFeatureSizeWarning):
            _ = ds[pf.feature_name]


def test_pf_wrong_length_2():
    """temporary feature should have same length"""
    h5path = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    with dclab.new_dataset(h5path) as ds:
        info = example_plugin_info_single_feature()
        info["method"] = lambda x: np.arange(len(ds)*2)
        pf = PlugInFeature("circ_per_area", info)
        with pytest.warns(BadFeatureSizeWarning):
            _ = ds[pf.feature_name]


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
            remove_all_plugin_features()
