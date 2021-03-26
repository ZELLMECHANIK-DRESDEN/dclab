"""Test command-line interface"""
import hashlib
import sys
import time

from dclab import cli, new_dataset, rtdc_dataset
import h5py
import imageio
import numpy as np
import pytest

from helper_methods import retrieve_data


def test_condense():
    path_in = retrieve_data("rtdc_data_hdf5_mask_contour.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("condensed.rtdc")

    cli.condense(path_out=path_out, path_in=path_in)
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-condense" in dsj.logs
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in dsj.features:
            assert np.all(dsj[feat] == ds0[feat])


def test_compress():
    path_in = retrieve_data("rtdc_data_hdf5_mask_contour.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("compressed.rtdc")

    cli.compress(path_out=path_out, path_in=path_in)
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-compress" in dsj.logs
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in ds0.features:
            if feat in ["contour", "image", "mask"]:
                for ii in range(len(dsj)):
                    assert np.all(dsj[feat][ii] == ds0[feat][ii]), feat
            else:
                assert np.all(dsj[feat] == ds0[feat]), feat


def test_compress_already_compressed():
    """By default, an already compressed dataset should not be compressed"""
    path_in = retrieve_data("rtdc_data_hdf5_mask_contour.zip")
    # same directory (will be cleaned up with path_in)
    path_out1 = path_in.with_name("compressed_1.rtdc")
    path_out2 = path_in.with_name("compressed_2.rtdc")
    path_out3 = path_in.with_name("compressed_copy_of_1.rtdc")
    # this is straight-forward
    cli.compress(path_out=path_out1, path_in=path_in)
    # just for the sake of comparison
    time.sleep(1)  # we need different time stamps in path_out2
    cli.compress(path_out=path_out2, path_in=path_in)
    # this is not trivial
    cli.compress(path_out=path_out3, path_in=path_out1)

    # the first two files should not be the same (dates are written, etc)
    h1 = hashlib.md5(path_out1.read_bytes()).hexdigest()
    h2 = hashlib.md5(path_out2.read_bytes()).hexdigest()
    h3 = hashlib.md5(path_out3.read_bytes()).hexdigest()
    assert h1 != h2
    assert h1 == h3


def test_compress_already_compressed_force():
    """An extension of the above test to make sure "force" works"""
    path_in = retrieve_data("rtdc_data_hdf5_mask_contour.zip")
    # same directory (will be cleaned up with path_in)
    path_out1 = path_in.with_name("compressed_1.rtdc")
    path_out2 = path_in.with_name("compressed_not_a_copy_of_1.rtdc")
    # this is straight-forward
    cli.compress(path_out=path_out1, path_in=path_in)
    # just for the sake of comparison
    cli.compress(path_out=path_out2, path_in=path_out1, force=True)

    # the first two files should not be the same (dates are written, etc)
    h1 = hashlib.md5(path_out1.read_bytes()).hexdigest()
    h2 = hashlib.md5(path_out2.read_bytes()).hexdigest()
    assert h1 != h2


def test_compress_correct_offset():
    """Testing if correction of offset works in compress-function"""
    # Arrange
    path_in = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    # Compress with and without offset correction
    path_out = path_in.with_name("compressed.rtdc")
    path_out_nc = path_in.with_name("compressed_nc.rtdc")

    with new_dataset(path_in) as ds:
        fl_max_old = ds["fl1_max"]
        fl_offset = min(np.min(ds["trace"]["fl1_raw"]), 1)

    # Act
    cli.compress(path_out=path_out, path_in=path_in, correct_offset=True)
    cli.compress(path_out=path_out_nc, path_in=path_in, correct_offset=False)

    # Assert
    with new_dataset(path_out) as ds:
        assert "baseline 1 offset" in ds.config["fluorescence"]
        assert fl_offset == ds.config["fluorescence"]["baseline 1 offset"]
        assert (ds["fl1_max"] == fl_max_old - (fl_offset-1)).all()

    with new_dataset(path_out_nc) as ds_nc:
        assert "baseline 1 offset" not in ds_nc.config["fluorescence"]
        assert (ds_nc["fl1_max"] == fl_max_old).all()


def test_join_tdms():
    path_in = retrieve_data("rtdc_data_shapein_v2.0.1.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.join(path_out=path_out, paths_in=[path_in, path_in])

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert len(dsj)
        assert len(dsj) == 2*len(ds0)
        assert len(ds0) == ds0.config["experiment"]["event count"]
        assert len(dsj) == dsj.config["experiment"]["event count"]
        assert np.all(dsj["circ"][:100] == ds0["circ"][:100])
        assert np.all(dsj["circ"][len(ds0):len(ds0)+100] == ds0["circ"][:100])
        assert set(dsj.features) == set(ds0.features)


def test_join_tdms_logs():
    path_in = retrieve_data("rtdc_data_shapein_v2.0.1.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.join(path_out=path_out, paths_in=[path_in, path_in])

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-join" in dsj.logs
        assert "cfg-#1" in dsj.logs
        assert "software version = ShapeIn 2.0.1" in dsj.logs["cfg-#1"]
        assert ds0.logs
        for key in ds0.logs:
            jkey = "src-#1_" + key
            assert np.all(np.array(ds0.logs[key]) == np.array(dsj.logs[jkey]))


def test_join_rtdc():
    path_in = retrieve_data("rtdc_data_hdf5_mask_contour.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.join(path_out=path_out, paths_in=[path_in, path_in])
    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-join" in dsj.logs
        assert len(dsj)
        assert len(dsj) == 2*len(ds0)
        assert np.all(dsj["circ"][:len(ds0)] == ds0["circ"])
        assert np.all(dsj["circ"][len(ds0):] == ds0["circ"])
        assert set(dsj.features) == set(ds0.features)
        assert 'identifier = ZMDD-AcC-8ecba5-cd57e2' in dsj.logs["cfg-#1"]


def test_join_times():
    path_in1 = retrieve_data("rtdc_data_hdf5_mask_contour.zip")
    path_in2 = retrieve_data("rtdc_data_hdf5_mask_contour.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in1.with_name("out.rtdc")

    # modify acquisition times
    with h5py.File(path_in1, mode="a") as h1:
        h1.attrs["experiment:date"] = "2019-11-04"
        h1.attrs["experiment:time"] = "15:00:00"

    with h5py.File(path_in2, mode="a") as h2:
        h2.attrs["experiment:date"] = "2019-11-05"
        h2.attrs["experiment:time"] = "16:01:15.050"

    offset = 24*60*60 + 60*60 + 1*60 + 15 + .05

    cli.join(path_out=path_out, paths_in=[path_in1, path_in2])
    with new_dataset(path_out) as dsj, new_dataset(path_in1) as ds0:
        assert np.allclose(dsj["time"],
                           np.concatenate((ds0["time"], ds0["time"]+offset)),
                           rtol=0,
                           atol=.0001)


def test_join_correct_offset():
    """Testing if correction of offset works in compress-function"""
    # Arrange
    path_in = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    # Join with and without offset correction
    path_out = path_in.with_name("compressed.rtdc")
    path_out_nc = path_in.with_name("compressed_nc.rtdc")

    with h5py.File(path_in, 'r') as hf:
        fl_max_old = hf["events"]["fl1_max"][:]
        fl_offset = min(np.min(hf["events"]["trace"]["fl1_raw"][:]), 1)

    # Act
    cli.join(path_out=path_out, paths_in=[path_in, path_in],
             correct_offset=True)
    cli.join(path_out=path_out_nc, paths_in=[path_in, path_in],
             correct_offset=False)

    # Assert
    with new_dataset(path_out) as ds:
        assert "baseline 1 offset" in ds.config["fluorescence"]
        assert fl_offset == ds.config["fluorescence"]["baseline 1 offset"]
        assert (ds["fl1_max"][:len(ds)//2] == fl_max_old - (fl_offset-1)).all()

    with new_dataset(path_out_nc) as ds_nc:
        assert "baseline 1 offset" not in ds_nc.config["fluorescence"]
        assert (ds_nc["fl1_max"][:len(ds_nc)//2] == fl_max_old).all()


def test_repack_basic():
    path_in = retrieve_data("rtdc_data_hdf5_mask_contour.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    cli.repack(path_out=path_out, path_in=path_in)

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in ds0.features_innate:
            if feat in ds0.features_scalar:
                assert np.all(dsj[feat] == ds0[feat]), feat
        for ii in range(len(ds0)):
            assert np.all(dsj["contour"][ii] == ds0["contour"][ii])
            assert np.all(dsj["image"][ii] == ds0["image"][ii])
            assert np.all(dsj["mask"][ii] == ds0["mask"][ii])


@pytest.mark.skipif(sys.version_info < (3, 6),
                    reason="requires python3.6 or higher")
def test_repack_remove_secrets():
    path_in = retrieve_data("rtdc_data_hdf5_mask_contour.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    with h5py.File(path_in, "a") as h5:
        h5.attrs["experiment:sample"] = "my dirty secret"

    with h5py.File(path_in, "a") as h5:
        h5.attrs["experiment:sample"] = "sunshine"

    # test whether the dirty secret is still there
    with open(str(path_in), "rb") as fd:
        data = fd.read()
        assert str(data).count("my dirty secret")

    # now repack
    cli.repack(path_out=path_out, path_in=path_in)

    # clean?
    with open(str(path_out), "rb") as fd:
        data = fd.read()
        assert not str(data).count("my dirty secret")


def test_repack_strip_logs():
    path_in = retrieve_data("rtdc_data_hdf5_mask_contour.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    # write some logs
    with h5py.File(path_in, "a") as h5:
        rtdc_dataset.write(h5,
                           logs={"test_log": ["peter", "hans"]},
                           mode="append")

    cli.repack(path_out=path_out, path_in=path_in, strip_logs=True)

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert ds0.logs
        assert not dsj.logs


def test_split():
    path_in = retrieve_data("rtdc_data_hdf5_mask_contour.zip")
    paths = cli.split(path_in=path_in, split_events=3, ret_out_paths=True)
    with new_dataset(path_in) as ds:
        ecount = 0
        for pp in paths:
            with new_dataset(pp) as di:
                for feat in ds.features_scalar:
                    if feat == "index":
                        continue
                    assert np.all(
                        ds[feat][ecount:ecount+len(di)] == di[feat]), feat
                ecount += len(di)


def test_split_correct_offset():
    # Arrange
    path_in = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    path_out = path_in.with_name("split")
    path_out_nc = path_in.with_name("split_nc")

    path_out.mkdir()
    path_out_nc.mkdir()

    with h5py.File(path_in, 'r') as hf:
        fl_offset = min(np.min(hf["events"]["trace"]["fl1_raw"][:]), 1)

    # Act
    paths = cli.split(path_in=path_in, path_out=path_out, split_events=3,
                      ret_out_paths=True, correct_offset=True)
    # do i need to time.sleep(1) here so the file names dont collide?
    paths_nc = cli.split(path_in=path_in, path_out=path_out_nc, split_events=3,
                         ret_out_paths=True, correct_offset=False)

    # Assert
    for path in paths:
        with new_dataset(path) as ds:
            assert "baseline 1 offset" in ds.config["fluorescence"]
            assert fl_offset == ds.config["fluorescence"]["baseline 1 offset"]

    for path in paths_nc:
        with new_dataset(path) as ds:
            assert "baseline 1 offset" not in ds.config["fluorescence"]


def test_split_traces():
    path_in = retrieve_data("rtdc_data_hdf5_rtfdc.zip")
    paths = cli.split(path_in=path_in, split_events=3, ret_out_paths=True)
    with new_dataset(path_in) as ds:
        ecount = 0
        for pp in paths:
            with new_dataset(pp) as di:
                for flkey in ds["trace"].keys():
                    trace1 = ds["trace"][flkey][ecount:ecount+len(di)]
                    trace2 = di["trace"][flkey][:]
                    assert len(trace1) == len(trace2)
                    assert np.all(trace1 == trace2), flkey
                ecount += len(di)


def test_tdms2rtdc():
    path_in = retrieve_data("rtdc_data_shapein_v2.0.1.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=False)

    with new_dataset(path_out) as ds2, new_dataset(path_in) as ds1:
        assert len(ds2)
        assert set(ds1.features) == set(ds2.features)
        # not all features are computed
        assert set(ds2._events.keys()) < set(ds1.features)
        for feat in ds1:
            assert np.all(ds1[feat] == ds2[feat])


def test_tdms2rtdc_features():
    path_in = retrieve_data("rtdc_data_shapein_v2.0.1.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=True)

    with new_dataset(path_out) as ds2, new_dataset(path_in) as ds1:
        assert len(ds2)
        assert set(ds1.features) == set(ds2.features)
        # features were computed
        assert set(ds2._events.keys()) == set(ds1.features)


def test_tdms2rtdc_remove_nan_image():
    path_in = retrieve_data("rtdc_data_traces_video_bright.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    # generate fake video
    with new_dataset(path_in) as ds:
        video_length = len(ds) - 1
    vname = path_in.with_name("M4_0.040000ul_s_imaq.avi")
    # remove contour data (not necessary for this test)
    path_in.with_name("M4_0.040000ul_s_contours.txt").unlink()

    imgs = imageio.mimread(vname)
    with imageio.get_writer(vname) as writer:
        for ii in range(video_length):
            writer.append_data(imgs[ii % len(imgs)])

    # without removal
    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=False,
                  skip_initial_empty_image=False)

    with new_dataset(path_out) as ds2, new_dataset(path_in) as ds1:
        assert len(ds2) == len(ds1)
        assert np.all(ds2["image"][0] == 0)

    # with removal
    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=False,
                  skip_initial_empty_image=True)

    with new_dataset(path_out) as ds2, new_dataset(path_in) as ds1:
        assert len(ds2) == video_length
        assert not np.all(ds2["image"][0] == 0)
        assert ds2.config["experiment"]["event count"] == video_length


def test_tdms2rtdc_update_roi_size():
    path_in = retrieve_data("rtdc_data_traces_video.zip")
    # set wrong roi sizes
    camin = path_in.with_name("M1_camera.ini")
    with camin.open("r") as fd:
        lines = fd.readlines()
    lines = lines[:-2]
    lines.append("width = 23\n")
    lines.append("height = 24\n")
    with camin.open("w") as fd:
        fd.writelines(lines)

    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=False,
                  skip_initial_empty_image=True)

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert ds0.config["imaging"]["roi size x"] == 23
        assert ds0.config["imaging"]["roi size y"] == 24
        assert dsj.config["imaging"]["roi size x"] == 256
        assert dsj.config["imaging"]["roi size y"] == 96
        wlog = "dclab-tdms2rtdc-warnings"
        assert "LimitingExportSizeWarning" in dsj.logs[wlog]


def test_tdms2rtdc_update_sample_per_events():
    path_in = retrieve_data("rtdc_data_traces_2flchan.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    # set wrong samples per event
    with path_in.with_name("M1_para.ini").open("a") as fd:
        fd.write("Samples Per Event = 1234")

    with new_dataset(path_in) as ds:
        assert ds.config["fluorescence"]["samples per event"] == 1234

    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=True)

    with new_dataset(path_out) as ds2:
        assert ds2.config["fluorescence"]["samples per event"] == 566


def test_tdms2rtdc_correct_offset():
    # Arrange
    path_in = retrieve_data("rtdc_data_traces_2flchan.zip")
    path_out = path_in.with_name("out.rtdc")
    path_out_nc = path_in.with_name("out_nc.rtdc")

    with new_dataset(path_in) as ds:
        fl_max_old = ds["fl2_max"]
        fl_offset = min(np.min(ds["trace"]["fl2_raw"]), 1)

    # Act
    cli.tdms2rtdc(path_tdms=path_in, path_rtdc=path_out,
                  correct_offset=True)
    cli.tdms2rtdc(path_tdms=path_in, path_rtdc=path_out_nc,
                  correct_offset=False)

    # Assert
    with new_dataset(path_out) as ds:
        assert fl_offset == ds.config["fluorescence"]["baseline 2 offset"]
        assert len(ds["fl2_max"]) == len(fl_max_old)
        assert (ds["fl2_max"] == fl_max_old - (fl_offset-1)).all()
        assert "baseline 2 offset" in ds.config["fluorescence"]

    with new_dataset(path_out_nc) as ds_nc:
        assert "baseline 2 offset" not in ds_nc.config["fluorescence"]
        assert (ds_nc["fl2_max"] == fl_max_old).all()


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
