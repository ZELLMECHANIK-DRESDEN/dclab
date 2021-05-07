"""command line interface"""
import argparse
import time
import hashlib
import json
import pathlib
import platform
import shutil
import sys
import warnings

import h5py
import numpy as np

from .rtdc_dataset import (
    check_dataset, export, fmt_tdms, new_dataset, write_hdf5)
from .rtdc_dataset.check import IntegrityChecker
from . import definitions as dfn
from . import util
from ._version import version

try:
    import imageio
except ModuleNotFoundError:
    imageio = None

try:
    import nptdms
except ModuleNotFoundError:
    nptdms = None


def assemble_warnings(w):
    """pretty-format all warnings for logs"""
    wlog = []
    for wi in w:
        wlog.append("{}".format(wi.category.__name__))
        wlog.append(" in {} line {}:".format(
            wi.category.__module__, wi.lineno))
        # make sure line is not longer than 100 characters
        words = str(wi.message).split(" ")
        wline = "  "
        for ii in range(len(words)):
            wline += " " + words[ii]
            if ii == len(words) - 1:
                # end
                wlog.append(wline)
            elif len(wline + words[ii+1]) + 1 >= 100:
                # new line
                wlog.append(wline)
                wline = "  "
            # nothing to do here
    return wlog


def get_command_log(paths, custom_dict=None):
    """Return a json dump of system parameters

    Parameters
    ----------
    paths: list of pathlib.Path or str
        paths of related measurement files; they are hashed
        and included in the "files" key
    custom_dict: dict
        additional user-defined entries; must contain simple
        Python objects (json.dumps must still work)
    """
    if custom_dict is None:
        custom_dict = {}
    data = get_job_info()
    data["files"] = []
    for ii, pp in enumerate(paths):
        fdict = {"name": pathlib.Path(pp).name,
                 "sha256": util.hashfile(pp, constructor=hashlib.sha256),
                 "index": ii+1
                 }
        data["files"].append(fdict)
    final_data = {}
    final_data.update(custom_dict)
    final_data.update(data)
    dump = json.dumps(final_data, sort_keys=True, indent=2).split("\n")
    return dump


def get_job_info():
    """Return dictionary with current job information

    Returns
    -------
    info: dict of dicts
        Job information including details about time, system,
        python version, and libraries used.
    """
    data = {
        "utc": {
            "date": time.strftime("%Y-%m-%d", time.gmtime()),
            "time": time.strftime("%H:%M:%S", time.gmtime()),
        },
        "system": {
            "info": platform.platform(),
            "machine": platform.machine(),
            "name": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": {
            "build": ", ".join(platform.python_build()),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "libraries": {
            "dclab": version,
            "h5py": h5py.__version__,
            "numpy": np.__version__,
        }
    }
    if imageio is not None:
        data["libraries"]["imageio"] = imageio.__version__
    if nptdms is not None:
        data["libraries"]["nptdms"] = nptdms.__version__
    return data


def print_info(string):
    print("\033[1m{}\033[0m".format(string))


def print_alert(string):
    print_info("\033[33m{}".format(string))


def print_violation(string):
    print_info("\033[31m{}".format(string))


def compress(path_out=None, path_in=None, force=False):
    """Create a new dataset with all features compressed losslessly"""
    if path_out is None or path_in is None:
        parser = compress_parser()
        args = parser.parse_args()
        path_in = args.input
        path_out = args.output
        force = args.force

    path_in = pathlib.Path(path_in)
    path_out = pathlib.Path(path_out)

    if not path_in.suffix == ".rtdc":
        raise ValueError("Please specify an .rtdc file!")

    if path_out.suffix != ".rtdc":
        path_out = path_out.with_name(path_out.name + ".rtdc")

    if not force:
        # Check whether the input file is already compressed
        # (This is not done in force-mode)
        ic = IntegrityChecker(path_in)
        cue = ic.check_compression()[0]
        if cue.data["uncompressed"] == 0:
            # we are done here
            shutil.copy2(path_in, path_out)  # copy with metadata
            return

    logs = {}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with new_dataset(path_in) as ds:
            ds.export.hdf5(path=path_out,
                           features=ds.features_innate,
                           filtered=False,
                           compression="gzip",
                           override=True)
            logs.update(ds.logs)

        # command log
        logs["dclab-compress"] = get_command_log(paths=[path_in])

        # warnings log
        if w:
            logs["dclab-compress-warnings"] = assemble_warnings(w)

    # Write log file
    with write_hdf5.write(path_out, logs=logs, mode="append",
                          compression="gzip"):
        pass


def compress_parser():
    descr = "Create a compressed version of an .rtdc file. This can be " \
            + "used for saving disk space (loss-less compression). The " \
            + "data generated during an experiment is usually not compressed."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('input', metavar="INPUT", type=str,
                        help='Input path (.rtdc file)')
    parser.add_argument('output', metavar="OUTPUT", type=str,
                        help='Output path (.rtdc file)')
    parser.add_argument('--force',
                        dest='force',
                        action='store_true',
                        help='Force compression, even if the input dataset '
                             + 'is already compressed.')
    parser.set_defaults(force=False)
    return parser


def condense(path_out=None, path_in=None):
    """Create a new dataset with all (ancillary) scalar-only features"""
    if path_out is None or path_in is None:
        parser = condense_parser()
        args = parser.parse_args()
        path_in = args.input
        path_out = args.output

    path_in = pathlib.Path(path_in)
    path_out = pathlib.Path(path_out)

    if path_out.suffix != ".rtdc":
        path_out = path_out.with_name(path_out.name + ".rtdc")

    logs = {}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with new_dataset(path_in) as ds:
            ds.export.hdf5(path=path_out,
                           features=ds.features_scalar,
                           filtered=False,
                           compression="gzip",
                           override=True)
            logs.update(ds.logs)

        # command log
        logs["dclab-condense"] = get_command_log(paths=[path_in])

        # warnings log
        if w:
            logs["dclab-condense-warnings"] = assemble_warnings(w)

    # Write log file
    with write_hdf5.write(path_out, logs=logs, mode="append",
                          compression="gzip"):
        pass


def condense_parser():
    descr = "Reduce an RT-DC measurement to its scalar-only features " \
            + "(i.e. without `contour`, `image`, `mask`, or `trace`). " \
            + "All available ancillary features are computed."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('input', metavar="INPUT", type=str,
                        help='Input path (.tdms or .rtdc file)')
    parser.add_argument('output', metavar="OUTPUT", type=str,
                        help='Output path (.rtdc file)')
    return parser


def join(path_out=None, paths_in=None, metadata=None):
    """Join multiple RT-DC measurements into a single .rtdc file"""
    if metadata is None:
        metadata = {"experiment": {"run index": 1}}
    if path_out is None or paths_in is None:
        parser = join_parser()
        args = parser.parse_args()
        paths_in = args.input
        path_out = args.output

    if len(paths_in) < 2:
        raise ValueError("At least two input files must be specified!")

    path_out = pathlib.Path(path_out)
    if not path_out.suffix == ".rtdc":
        path_out = path_out.parent / (path_out.name + ".rtdc")

    if path_out.exists():
        raise ValueError("Output file '{}' already exists!".format(path_out))

    # Determine input file order
    key_paths = []
    for pp in paths_in:
        with new_dataset(pp) as ds:
            key = "_".join([ds.config["experiment"]["date"],
                            ds.config["experiment"]["time"],
                            str(ds.config["experiment"]["run index"])
                            ])
            key_paths.append((key, pp))
    sorted_paths = [p[1] for p in sorted(key_paths, key=lambda x: x[0])]

    # Determine temporal offsets
    toffsets = np.zeros(len(sorted_paths), dtype=float)
    for ii, pp in enumerate(sorted_paths):
        with new_dataset(pp) as ds:
            etime = ds.config["experiment"]["time"]
            st = time.strptime(ds.config["experiment"]["date"]
                               + etime[:8],
                               "%Y-%m-%d%H:%M:%S")
            toffsets[ii] = time.mktime(st)
            if len(etime) > 8:
                # floating point time stored as well (HH:MM:SS.SS)
                toffsets[ii] += float(etime[8:])
    toffsets -= toffsets[0]

    logs = {}
    # Create initial output file
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with new_dataset(sorted_paths[0]) as ds0:
            features = sorted(ds0.features_innate)
            ds0.export.hdf5(path=path_out,
                            features=features,
                            filtered=False,
                            override=True,
                            compression="gzip")
        if w:
            logs["dclab-join-warnings-#1"] = assemble_warnings(w)

    with write_hdf5.write(path_out, mode="append",
                          compression="gzip") as h5obj:
        ii = 1
        # Append data from other files
        for pi, ti in zip(sorted_paths[1:], toffsets[1:]):
            ii += 1  # we start with the second dataset
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with new_dataset(pi) as dsi:
                    for feat in features:
                        export.hdf5_append(h5obj=h5obj,
                                           rtdc_ds=dsi,
                                           feat=feat,
                                           compression="gzip",
                                           time_offset=ti)
                if w:
                    lkey = "dclab-join-warnings-#{}".format(ii)
                    logs[lkey] = assemble_warnings(w)
        export.hdf5_autocomplete_config(h5obj)

    # Logs and configs from source files
    logs["dclab-join"] = get_command_log(paths=sorted_paths)
    for ii, pp in enumerate(sorted_paths):
        with new_dataset(pp) as ds:
            # data file logs
            for ll in ds.logs:
                logs["src-#{}_{}".format(ii+1, ll)] = ds.logs[ll]
            # configuration
            cfg = ds.config.tostring(sections=dfn.CFG_METADATA).split("\n")
            logs["cfg-#{}".format(ii+1)] = cfg

    # Write logs and missing meta data
    with write_hdf5.write(path_out, logs=logs, meta=metadata, mode="append",
                          compression="gzip"):
        pass


def join_parser():
    descr = "Join two or more RT-DC measurements. This will produce " \
            + "one larger .rtdc file. The meta data of the dataset " \
            + "that was recorded earliest will be used in the output " \
            + "file. Please only join datasets that were recorded " \
            + "in the same measurement run."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('input', metavar="INPUT", nargs="*", type=str,
                        help='Input paths (.tdms or .rtdc files)')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-o', '--output', metavar="OUTPUT", type=str,
                                help='Output path (.rtdc file)', required=True)
    return parser


def repack(path_in=None, path_out=None, strip_logs=False):
    """Repack/recreate an .rtdc file, optionally stripping the logs"""
    if path_in is None and path_out is None:
        parser = repack_parser()
        args = parser.parse_args()
        path_in = args.input
        path_out = args.output
        strip_logs = args.strip_logs

    with new_dataset(path_in) as ds, h5py.File(path_out, "w") as h5:
        # write metadata first (to avoid resetting software version)
        # only export configuration meta data (no user-defined config)
        meta = {}
        for sec in dfn.CFG_METADATA:
            if sec in ds.config:
                meta[sec] = ds.config[sec].copy()

        write_hdf5.write(h5, meta=meta, mode="append")

        if not strip_logs:
            write_hdf5.write(h5, logs=ds.logs, mode="append")

        # write features
        for feat in ds.features_innate:
            export.hdf5_append(h5obj=h5,
                               rtdc_ds=ds,
                               feat=feat,
                               compression="gzip",
                               filtarr=None,
                               time_offset=0)


def repack_parser():
    descr = "Repack an .rtdc file. The difference to dclab-compress " \
            + "is that no logs are added. Other logs can optionally be " \
            + "stripped away. Repacking also gets rid of old clutter " \
            + "data (e.g. previous metadata stored in the HDF5 file)."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('input', metavar="INPUT", type=str,
                        help='Input path (.rtdc file)')
    parser.add_argument('output',  metavar="OUTPUT", type=str,
                        help='Output path (.rtdc file)')
    parser.add_argument('--strip-logs',
                        dest='strip_logs',
                        action='store_true',
                        help='Do not copy any logs to the output file.')
    parser.set_defaults(strip_logs=False)
    return parser


def skip_empty_image_events(ds, initial=True, final=True):
    """Set a manual filter to skip inital or final empty image events"""
    if initial:
        if (("image" in ds and ds.format == "tdms"
             and ds.config["fmt_tdms"]["video frame offset"])
            or ("contour" in ds and np.all(ds["contour"][0] == 0))
                or ("image" in ds and np.all(ds["image"][0] == 0))):
            ds.filter.manual[0] = False
            ds.apply_filter()
    if final:
        # This is not easy to test, because we need a corrupt
        # frame.
        if "image" in ds:
            idfin = len(ds) - 1
            if ds.format == "tdms":
                with warnings.catch_warnings(record=True) as wfin:
                    warnings.simplefilter(
                        "always",
                        fmt_tdms.event_image.CorruptFrameWarning)
                    _ = ds["image"][idfin]  # provoke a warning
                    if wfin:
                        ds.filter.manual[idfin] = False
                        ds.apply_filter()
            elif np.all(ds["image"][idfin] == 0):
                ds.filter.manual[idfin] = False
                ds.apply_filter()


def split(path_in=None, path_out=None, split_events=10000,
          skip_initial_empty_image=True, skip_final_empty_image=True,
          ret_out_paths=False, verbose=False):
    """Split a measurement file

    Parameters
    ----------
    path_in: str or pathlib.Path
        Path of input measurement file
    path_out: str or pathlib.Path
        Path to output directory (optional)
    split_events: int
        Maximum number of events in each output file
    skip_initial_empty_image: bool
        Remove the first event of the dataset if the image is zero.
    skip_final_empty_image: bool
        Remove the final event of the dataset if the image is zero.
    ret_out_paths:
        If True, return the list of output file paths.
    verbose: bool
        If `True`, print messages to stdout

    Returns
    -------
    [out_paths]: list of pathlib.Path
        List of generated files (only if `ret_out_paths` is specified)
    """
    if path_in is None:
        parser = split_parser()
        args = parser.parse_args()

        path_in = pathlib.Path(args.path_in).resolve()
        path_out = args.path_out
        split_events = args.split_events
        skip_initial_empty_image = not args.include_empty_boundary_images
        skip_final_empty_image = not args.include_empty_boundary_images
        verbose = True
    if path_out in ["PATH_IN", None]:  # default to input directory
        path_out = path_in.parent
    path_in = pathlib.Path(path_in)
    path_out = pathlib.Path(path_out)
    logs = {"dclab-split": get_command_log(paths=[path_in])}
    paths_gen = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # ignore ResourceWarning: unclosed file <_io.BufferedReader...>
        warnings.simplefilter("ignore", ResourceWarning)  # noqa: F821
        # ignore SlowVideoWarning
        warnings.simplefilter("ignore",
                              fmt_tdms.event_image.SlowVideoWarning)
        if skip_initial_empty_image:
            # If the initial frame is skipped when empty,
            # suppress any related warning messages.
            warnings.simplefilter(
                "ignore",
                fmt_tdms.event_image.InitialFrameMissingWarning)

        with new_dataset(path_in) as ds:
            for ll in ds.logs:
                logs["src-{}".format(ll)] = ds.logs[ll]
            num_files = len(ds) // split_events
            if 10 % 4:
                num_files += 1
            for ii in range(num_files):
                pp = path_out / "{}_{:04d}.rtdc".format(path_in.stem, ii+1)
                paths_gen.append(pp)
                if verbose:
                    print(
                        "Generating {:d}/{:d}: {}".format(ii+1, num_files, pp))
                ds.filter.manual[:] = False  # reset filter
                ds.filter.manual[ii*split_events:(ii+1)*split_events] = True
                skip_empty_image_events(ds, initial=skip_initial_empty_image,
                                        final=skip_final_empty_image)
                ds.apply_filter()
                ds.export.hdf5(
                    path=pp, features=ds.features_innate, filtered=True)
        if w:
            logs["dclab-split-warnings"] = assemble_warnings(w)
        sample_name = ds.config["experiment"]["sample"]

    # Add the logs and update sample name
    for ii, pp in enumerate(paths_gen):
        meta = {"experiment": {"sample": "{} {}/{}".format(sample_name,
                                                           ii+1,
                                                           num_files)}}
        with write_hdf5.write(pp, logs=logs, meta=meta, mode="append",
                              compression="gzip"):
            pass

    if ret_out_paths:
        return paths_gen


def split_parser():
    descr = "Split an RT-DC measurement file (.tdms or .rtdc) into multiple " \
            + "smaller .rtdc files."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('path_in', metavar="PATH_IN", type=str,
                        help='Input path (.tdms or .rtdc file)')
    parser.add_argument('--path_out', metavar="PATH_OUT", type=str,
                        default="PATH_IN",
                        help='Output directory (defaults to same directory)')
    parser.add_argument('--split-events', type=int, default=10000,
                        help='Maximum number of events in each output file')
    parser.add_argument('--include-empty-boundary-images',
                        dest='include_empty_boundary_images',
                        action='store_true',
                        help='In old versions of Shape-In, the first or last '
                             + 'images were sometimes not stored in the '
                             + 'resulting .avi file. In dclab, such images '
                             + 'are represented as zero-valued images. Set '
                             + 'this option, if you wish to include these '
                             + 'events with empty image data.')
    parser.set_defaults(include_empty_boundary_images=False)

    return parser


def tdms2rtdc(path_tdms=None, path_rtdc=None, compute_features=False,
              skip_initial_empty_image=True, skip_final_empty_image=True,
              verbose=False):
    """Convert .tdms datasets to the hdf5-based .rtdc file format

    Parameters
    ----------
    path_tdms: str or pathlib.Path
        Path to input .tdms file
    path_rtdc: str or pathlib.Path
        Path to output .rtdc file
    compute_features: bool
        If `True`, compute all ancillary features and store them in the
        output file
    skip_initial_empty_image: bool
        In old versions of Shape-In, the first image was sometimes
        not stored in the resulting .avi file. In dclab, such images
        are represented as zero-valued images. If `True` (default),
        this first image is not included in the resulting .rtdc file.
    skip_final_empty_image: bool
        In other versions of Shape-In, the final image is sometimes
        also not stored in the .avi file. If `True` (default), this
        final image is not included in the resulting .rtdc file.
    verbose: bool
        If `True`, print messages to stdout
    """
    if path_tdms is None or path_rtdc is None:
        parser = tdms2rtdc_parser()
        args = parser.parse_args()

        path_tdms = pathlib.Path(args.tdms_path).resolve()
        path_rtdc = pathlib.Path(args.rtdc_path)
        compute_features = args.compute_features
        skip_initial_empty_image = not args.include_empty_boundary_images
        skip_final_empty_image = not args.include_empty_boundary_images
        verbose = True

    if not path_tdms.suffix == ".tdms":
        raise ValueError("Please specify a .tdms file!")

    if not path_rtdc.suffix == ".rtdc":
        path_rtdc = path_rtdc.with_name(path_rtdc.name + ".rtdc")

    # Determine whether input path is a tdms file or a directory
    if path_tdms.is_dir():
        files_tdms = fmt_tdms.get_tdms_files(path_tdms)
        if path_rtdc.is_file():
            raise ValueError("rtdc_path is a file: {}".format(path_rtdc))
        files_rtdc = []
        for ff in files_tdms:
            ff = pathlib.Path(ff)
            rp = ff.relative_to(path_tdms)
            # determine output file name (same relative path)
            rpr = path_rtdc / rp.with_suffix(".rtdc")
            files_rtdc.append(rpr)
    else:
        files_tdms = [path_tdms]
        files_rtdc = [path_rtdc]

    for ii in range(len(files_tdms)):
        ff = pathlib.Path(files_tdms[ii])
        fr = pathlib.Path(files_rtdc[ii])

        if verbose:
            print_info("Converting {:d}/{:d}: {}".format(
                ii + 1, len(files_tdms), ff))
        # create directory
        if not fr.parent.exists():
            fr.parent.mkdir(parents=True)
        # load and export dataset
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # ignore ResourceWarning: unclosed file <_io.BufferedReader...>
            warnings.simplefilter("ignore", ResourceWarning)  # noqa: F821
            # ignore SlowVideoWarning
            warnings.simplefilter("ignore",
                                  fmt_tdms.event_image.SlowVideoWarning)
            if skip_initial_empty_image:
                # If the initial frame is skipped when empty,
                # suppress any related warning messages.
                warnings.simplefilter(
                    "ignore",
                    fmt_tdms.event_image.InitialFrameMissingWarning)

            with new_dataset(ff) as ds:
                # determine features to export
                if compute_features:
                    features = ds.features
                else:
                    # consider special case for "image", "trace", and "contour"
                    # (This will export both "mask" and "contour".
                    # The "mask" is computed from "contour" and it is needed
                    # by dclab for other ancillary features. We still keep
                    # "contour" because it is original data.
                    features = ds.features_innate

                skip_empty_image_events(ds, initial=skip_initial_empty_image,
                                        final=skip_final_empty_image)
                # export as hdf5
                ds.export.hdf5(path=fr,
                               features=features,
                               filtered=True,
                               override=True,
                               compression="gzip")

                # write logs
                custom_dict = {}
                # computed features
                cfeats = list(set(features) - set(ds.features_innate))
                if "mask" in features:
                    # Mask is always computed from contour data
                    cfeats.append("mask")
                custom_dict["ancillary features"] = sorted(cfeats)

                # command log
                logs = {"dclab-tdms2rtdc": get_command_log(
                    paths=[ff], custom_dict=custom_dict)}
                # warnings log
                if w:
                    logs["dclab-tdms2rtdc-warnings"] = assemble_warnings(w)
                logs.update(ds.logs)
                with write_hdf5.write(fr, logs=logs, mode="append",
                                      compression="gzip"):
                    pass


def tdms2rtdc_parser():
    descr = "Convert RT-DC .tdms files to the hdf5-based .rtdc file format. " \
            + "Note: Do not delete original .tdms files after conversion. " \
            + "The conversion might be incomplete."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('--compute-ancillary-features',
                        dest='compute_features',
                        action='store_true',
                        help='Compute features, such as volume or emodulus, '
                             + 'that are otherwise computed on-the-fly. '
                             + 'Use this if you want to minimize analysis '
                             + 'time in e.g. Shape-Out. CAUTION: ancillary '
                             + 'feature recipes might be subject to change '
                             + '(e.g. if an error is found in the recipe). '
                             + 'Disabling this option maximizes '
                             + 'compatibility with future versions and '
                             + 'allows to isolate the original data.')
    parser.set_defaults(compute_features=False)
    parser.add_argument('--include-empty-boundary-images',
                        dest='include_empty_boundary_images',
                        action='store_true',
                        help='In old versions of Shape-In, the first or last '
                             + 'images were sometimes not stored in the '
                             + 'resulting .avi file. In dclab, such images '
                             + 'are represented as zero-valued images. Set '
                             + 'this option, if you wish to include these '
                             + 'events with empty image data.')
    parser.set_defaults(include_empty_boundary_images=False)
    parser.add_argument('tdms_path', metavar="TDMS_PATH", type=str,
                        help='Input path (tdms file or folder containing '
                             + 'tdms files)')
    parser.add_argument('rtdc_path', metavar="RTDC_PATH", type=str,
                        help='Output path (file or folder), existing data '
                             + 'will be overridden')
    return parser


def verify_dataset():
    """Perform checks on experimental datasets"""
    parser = verify_dataset_parser()
    args = parser.parse_args()
    path_in = pathlib.Path(args.path).resolve()
    print_info("Checking {}".format(path_in))
    # The exit status of this script. Non-zero exit status means:
    # 1: alerts
    # 2: violations
    # 3: alerts and violations
    # 4: other error
    exit_status = 4
    try:
        viol, aler, info = check_dataset(path_in)
    except fmt_tdms.InvalidTDMSFileFormatError:
        print_violation("Invalid tdms file format!")
    except fmt_tdms.IncompleteTDMSFileFormatError:
        print_violation("Incomplete dataset!")
    except fmt_tdms.ContourIndexingError:
        print_violation("Invalid contour data!")
    except fmt_tdms.InvalidVideoFileError:
        print_violation("Invalid image data!")
    except BaseException as e:
        print_violation(", ".join(e.args))
    else:
        for inf in info:
            print_info(inf)
        for ale in aler:
            print_alert(ale)
        for vio in viol:
            print_violation(vio)
        print_info("Check Complete: {} violations and ".format(len(viol))
                   + "{} alerts".format(len(aler)))
        if aler and viol:
            exit_status = 3
        elif aler:
            exit_status = 1
        elif viol:
            exit_status = 2
        else:
            # everything is ok
            exit_status = 0
    finally:
        sys.exit(exit_status)


def verify_dataset_parser():
    descr = "Check experimental datasets for completeness. This command " \
            + "is used e.g. to enforce data integrity with Shape-In. The " \
            + "following exit codes are defined: ``0: valid dataset``, " \
            + "``1: alerts encountered``, ``2: violations encountered``, " \
            + "``3: alerts and violations``, ``4: other error``."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('path', metavar='PATH', type=str,
                        help='Path to experimental dataset')
    return parser
