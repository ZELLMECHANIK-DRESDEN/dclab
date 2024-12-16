from __future__ import annotations

import atexit
from contextlib import ExitStack
import copy
import functools
import json
import pathlib
# replace this import when dropping support for Python 3.8
# from importlib import resources as importlib_resources
import importlib_resources
import warnings

import numpy as np
import numpy.typing as npt

from ... import definitions as dfn


_file_manager = ExitStack()
atexit.register(_file_manager.close)

#: Dictionary of look-up tables that the user added via :func:`register_lut`.
EXTERNAL_LUTS = {}


@functools.lru_cache()
def get_internal_lut_names_dict() -> dict:
    """Return list of internal lut names"""
    lutfiles = {}
    for pp in importlib_resources.files('dclab.features.emodulus').iterdir():
        name = pp.name
        if name.startswith("lut_") and name.endswith(".txt"):
            lutfiles[name[4:-4]] = name
    return lutfiles


def get_lut_path(path_or_id: str | pathlib.Path) -> pathlib.Path:
    """Find the path to a LUT

    path_or_id: str or pathlib.Path
        Identifier of a LUT. This can be either an existing path
        (checked first), or an internal identifier (see
        :func:`get_internal_lut_names_dict`).
    """
    internal_dict = get_internal_lut_names_dict()
    if path_or_id == "FEM-2Daxis":
        # backwards compatibility
        warnings.warn("'FEM-2Daxis' is deprecated. Please use 'LE-2D-FEM-19'!",
                      DeprecationWarning)
        path_or_id = "LE-2D-FEM-19"
    if pathlib.Path(path_or_id).exists():
        lut_path = pathlib.Path(path_or_id)
    elif path_or_id in internal_dict:
        ref = importlib_resources.files(
            "dclab.features.emodulus") / internal_dict[path_or_id]
        lut_path = _file_manager.enter_context(
            importlib_resources.as_file(ref))
    elif path_or_id in EXTERNAL_LUTS:
        lut_path = EXTERNAL_LUTS[path_or_id]
    else:
        raise ValueError("File or LUT identifier does not exist: "
                         + "'{}'".format(path_or_id))
    return lut_path


def load_lut(lut_data: str | pathlib.Path | tuple[npt.NDArray, dict] =
             "LE-2D-FEM-19") -> tuple[npt.NDArray, dict]:
    """Load LUT data from disk

    Parameters
    ----------
    lut_data: path, str, or tuple of (np.ndarray of shape (N, 3), dict)
        The LUT data to use. If it is in :func:`get_internal_lut_names_dict()`,
        then the respective LUT will be used. Otherwise, a path to a
        file on disk or a tuple (LUT array, metadata) is possible.

    Returns
    -------
    lut: np.ndarray of shape (N, 3)
        The LUT data for interpolation
    meta: dict
        The LUT metadata

    Notes
    -----
    If lut_data is a tuple of (lut, meta), then nothing is actually
    done (this is implemented for user convenience).
    """
    if isinstance(lut_data, tuple):
        lut, meta = lut_data
        lut = np.array(lut, copy=True)  # copy, because of normalization
        meta = copy.deepcopy(meta)  # copy, for the sake of consistency
    elif isinstance(lut_data, (str, pathlib.Path)):
        lut_path = get_lut_path(lut_data)
        lut, meta = load_mtext(lut_path)
    else:
        raise ValueError("`name_path_arr` must be path, identifier, or tuple "
                         f"of (array, dict), got '{lut_data}'!")
    return lut, meta


def load_mtext(path: str | pathlib.Path) -> tuple[npt.NDArray, dict]:
    """Load column-based data from text files with metadata

    This file format is used for isoelasticity lines and look-up
    table data in dclab.

    The text file is loaded with `numpy.loadtxt`. The metadata
    are stored as a json string between the "BEGIN METADATA" and
    the "END METADATA" tags. The last comment (#) line before the
    actual data defines the features with units in square
    brackets and tab-separated. For instance:

        # [...]
        #
        # BEGIN METADATA
        # {
        #   "authors": "A. Mietke, C. Herold, J. Guck",
        #   "channel_width": 20.0,
        #   "channel_width_unit": "um",
        #   "date": "2018-01-30",
        #   "dimensionality": "2Daxis",
        #   "flow_rate": 0.04,
        #   "flow_rate_unit": "uL/s",
        #   "fluid_viscosity": 15.0,
        #   "fluid_viscosity_unit": "mPa s",
        #   "identifier": "LE-2D-ana-18",
        #   "method": "analytical",
        #   "model": "linear elastic",
        #   "publication": "https://doi.org/10.1016/j.bpj.2015.09.006",
        #   "software": "custom Matlab code",
        #   "summary": "2D-axis-symmetric analytical solution"
        # }
        # END METADATA
        #
        # [...]
        #
        # area_um [um^2]    deform    emodulus [kPa]
        3.75331e+00    5.14496e-03    9.30000e-01
        4.90368e+00    6.72683e-03    9.30000e-01
        6.05279e+00    8.30946e-03    9.30000e-01
        7.20064e+00    9.89298e-03    9.30000e-01
        [...]
    """
    path = pathlib.Path(path).resolve()

    # Parse metadata
    size = path.stat().st_size
    dump = []
    injson = False
    prev_line = ""
    with path.open("r", errors='replace') as fd:
        while True:
            line = fd.readline()
            if fd.tell() == size:
                # something went wrong
                raise ValueError("EOF: Could not parse '{}'!".format(path))
            elif len(line.strip()) == 0:
                # ignore empty lines
                continue
            elif not line.strip().startswith("#"):
                # we are done here
                if prev_line == "":
                    raise ValueError("No column header in '{}'!".format(
                        path))
                break
            elif line.startswith("# BEGIN METADATA"):
                injson = True
                continue
            elif line.startswith("# END METADATA"):
                injson = False
            if injson:
                dump.append(line.strip("#").strip())
            else:
                # remember last line for header
                prev_line = line
    # metadata
    if dump:
        meta = json.loads("\n".join(dump))
    else:
        raise ValueError("No metadata json dump in '{}'!".format(path))
    # header
    feats = []
    units = []
    for hh in prev_line.strip("# ").split("\t"):
        if hh.count(" "):
            ft, un = hh.strip().split(" ")
            un = un.strip("[]")
        else:
            ft = hh
            un = ""
        if not dfn.scalar_feature_exists(ft):
            raise ValueError("Scalar feature not known: '{}'".format(ft))
        feats.append(ft)
        units.append(un)
    # data
    data = np.loadtxt(path)

    meta["column features"] = feats
    meta["column units"] = units

    # sanity checks
    assert meta["channel_width_unit"] == "um"
    assert meta["flow_rate_unit"] == "uL/s"
    assert meta["fluid_viscosity_unit"] == "mPa s"
    for ft, un in zip(feats, units):
        if ft == "deform":
            assert un == ""
        elif ft == "area_um":
            assert un == "um^2"
        elif ft == "emodulus":
            assert un == "kPa"
        elif ft == "volume":
            assert un == "um^3"
        else:
            assert False, "Please add sanity check for {}!".format(ft)

    return data, meta


def register_lut(path: str | pathlib.Path, identifier: str = None) -> None:
    """Register an external LUT file in dclab

    This will add it to :const:`EXTERNAL_LUTS`, which is required
    for emodulus computation as an ancillary feature.

    Parameters
    ----------
    path: str or pathlib.Path
        Path to the external LUT file
    identifier: str or None
        The identifier is used for ancillary emodulus computation
        via the [calculation]: "emodulus lut" key. It is also used as
        the key in :const:`EXTERNAL_LUTS` during registration. If not
        specified, (default) then the identifier given as JSON metadata
        in `path` is used.
    """
    if identifier is None:
        _, md = load_mtext(path)
        try:
            identifier = md["identifier"]
        except KeyError:
            raise ValueError("The given LUT file '{}' does ".format(path)
                             + "not contain the 'identifier' keyword. You may "
                             + "specify it via the `identifier` keyword to "
                             + "this function.")
    if identifier in EXTERNAL_LUTS:
        raise ValueError("A LUT with an identifier '{}' ".format(identifier)
                         + "has already been registered!")
    elif identifier in get_internal_lut_names_dict():
        raise ValueError("The identifier '{}' is already ".format(identifier)
                         + "in use by an internal LUT!")
    EXTERNAL_LUTS[identifier] = path
