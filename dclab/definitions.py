"""Naming conventions"""
import copy
import numpy as np

from .rtdc_dataset.ancillaries import AncillaryFeature
from .parse_funcs import fbool, fint, fintlist, func_types, lcstr


#: All configuration keywords editable by the user
CFG_ANALYSIS = {
    # filtering parameters
    "filtering": [
        ["hierarchy parent", str, "Hierarchy parent of the dataset"],
        ["remove invalid events", fbool, "Remove events with inf/nan values"],
        ["enable filters", fbool, "Enable filtering"],
        ["limit events", fint, "Upper limit for number of filtered events"],
        ["polygon filters", fintlist, "Polygon filter indices"],
    ],
    # Addition user-defined data
    "calculation": [
        # "emodulus lut" was introduced in 0.32.0 and will replace
        # the deprecated "emodulus model".
        ["emodulus lut", str, "Look-up table identifier"],
        ["emodulus model", lcstr, "Model [DEPRECATED]"],
        ["emodulus medium", str, "Medium used (e.g. CellCarrierB, water)"],
        ["emodulus temperature", float, "Chip temperature [°C]"],
        ["emodulus viscosity", float, "Viscosity [Pa*s] if 'medium' unknown"],
        ["crosstalk fl21", float, "Fluorescence crosstalk, channel 2 to 1"],
        ["crosstalk fl31", float, "Fluorescence crosstalk, channel 3 to 1"],
        ["crosstalk fl12", float, "Fluorescence crosstalk, channel 1 to 2"],
        ["crosstalk fl32", float, "Fluorescence crosstalk, channel 3 to 2"],
        ["crosstalk fl13", float, "Fluorescence crosstalk, channel 1 to 3"],
        ["crosstalk fl23", float, "Fluorescence crosstalk, channel 2 to 3"],
    ]
}

#: All read-only configuration keywords for a measurement
CFG_METADATA = {
    # All parameters related to the actual experiment
    "experiment": [
        ["date", str, "Date of measurement ('YYYY-MM-DD')"],
        ["event count", fint, "Number of recorded events"],
        ["run index", fint, "Index of measurement run"],
        ["sample", str, "Measured sample or user-defined reference"],
        ["time", str, "Start time of measurement ('HH:MM:SS[.S]')"],
    ],
    # All special keywords related to RT-FDC
    # This section should not be present for regular RT-DC measurements.
    "fluorescence": [
        # The baseline offset was introduced in 0.33.0. It is added to
        # the trace data to obtain the actual signal used for data
        # processing (e.g. obtaining the fl1_max feature). This is more
        # robust than adding the offset directly to the trace data, because
        # of the possibility of integer overflows. Furthermore, DCKit can
        # set this parameter without modifying the original trace data
        # to correct/remove negative trace data
        # (see https://github.com/ZELLMECHANIK-DRESDEN/dclab/issues/101).
        ["baseline 1 offset", fint, "Baseline offset channel 1"],
        ["baseline 2 offset", fint, "Baseline offset channel 2"],
        ["baseline 3 offset", fint, "Baseline offset channel 3"],
        ["bit depth", fint, "Trace bit depth"],
        # If a fluorescence channel is used, a channel name *must* be
        # present. If a channel is not used, the channel name *must not*
        # be present. E.g. if only channels 1 and 2 are used, but there
        # are three channels present, then `channel count` is two,
        # `channels installed` is three, and `channel 3 name` is not set.
        ["channel 1 name", str, "FL1 description"],
        ["channel 2 name", str, "FL2 description"],
        ["channel 3 name", str, "FL3 description"],
        ["channel count", fint, "Number of active channels"],
        ["channels installed", fint, "Number of available channels"],
        # In contrast to `channel ? name`, the laser power *may*
        # be present (but must be set to 0), if a laser line is not used.
        ["laser 1 lambda", float, "Laser 1 wavelength [nm]"],
        ["laser 1 power", float, "Laser 1 output power [%]"],
        ["laser 2 lambda", float, "Laser 2 wavelength [nm]"],
        ["laser 2 power", float, "Laser 2 output power [%]"],
        ["laser 3 lambda", float, "Laser 3 wavelength [nm]"],
        ["laser 3 power", float, "Laser 3 output power [%]"],
        ["laser count", fint, "Number of active lasers"],
        ["lasers installed", fint, "Number of available lasers"],
        ["sample rate", float, "Trace sample rate [Hz]"],
        ["samples per event", fint, "Samples per event"],
        ["signal max", float, "Upper voltage detection limit [V]"],
        ["signal min", float, "Lower voltage detection limit [V]"],
        ["trace median", fint, "Rolling median filter size for traces"],
    ],
    # All tdms-related parameters
    "fmt_tdms": [
        ["video frame offset", fint, "Missing events at beginning of video"],
    ],
    # All imaging-related keywords
    "imaging": [
        ["flash device", str, "Light source device type"],  # e.g. green LED
        ["flash duration", float, "Light source flash duration [µs]"],
        ["frame rate", float, "Imaging frame rate [Hz]"],
        ["pixel size", float, "Pixel size [µm]"],
        ["roi position x", float, "Image x coordinate on sensor [px]"],
        ["roi position y", float, "Image y coordinate on sensor [px]"],
        ["roi size x", fint, "Image width [px]"],
        ["roi size y", fint, "Image height [px]"],
    ],
    # All parameters for online contour extraction from the event images
    "online_contour": [
        ["bin area min", fint, "Minium pixel area of binary image event"],
        ["bin kernel", fint, "Odd ellipse kernel size, binary image morphing"],
        ["bin threshold", fint, "Binary threshold for avg-bg-corrected image"],
        ["image blur", fint, "Odd sigma for Gaussian blur (21x21 kernel)"],
        ["no absdiff", fbool, "Avoid OpenCV 'absdiff' for avg-bg-correction"],
    ],
    # All online filters
    "online_filter": [
        ["area_ratio max", float, "Maximum porosity"],
        ["area_ratio min", float, "Minimum porosity"],
        ["area_ratio soft limit", fbool, "Soft limit, porosity"],
        ["area_um max", float, "Maximum area [µm²]"],
        ["area_um min", float, "Minimum area [µm²]"],
        ["area_um soft limit", fbool, "Soft limit, area [µm²]"],
        ["aspect max", float, "Maximum aspect ratio of bounding box"],
        ["aspect min", float, "Minimum aspect ratio of bounding box"],
        ["aspect soft limit", fbool, "Soft limit, aspect ratio of bbox"],
        ["deform max", float, "Maximum deformation"],
        ["deform min", float, "Minimum deformation"],
        ["deform soft limit", fbool, "Soft limit, deformation"],
        ["fl1_max max", float, "Maximum FL-1 maximum [a.u.]"],
        ["fl1_max min", float, "Minimum FL-1 maximum [a.u.]"],
        ["fl1_max soft limit", fbool, "Soft limit, FL-1 maximum"],
        ["fl2_max max", float, "Maximum FL-2 maximum [a.u.]"],
        ["fl2_max min", float, "Minimum FL-2 maximum [a.u.]"],
        ["fl2_max soft limit", fbool, "Soft limit, FL-2 maximum"],
        ["fl3_max max", float, "Maximum FL-3 maximum [a.u.]"],
        ["fl3_max min", float, "Minimum FL-3 maximum [a.u.]"],
        ["fl3_max soft limit", fbool, "Soft limit, FL-3 maximum"],
        ["size_x max", fint, "Maximum bounding box size x [µm]"],
        ["size_x min", fint, "Minimum bounding box size x [µm]"],
        ["size_x soft limit", fbool, "Soft limit, bounding box size x"],
        ["size_y max", fint, "Maximum bounding box size y [µm]"],
        ["size_y min", fint, "Minimum bounding box size y [µm]"],
        ["size_y soft limit", fbool, "Soft limit, bounding box size y"],
        # "target*" is only set if measurement is stopped automatically.
        # "target*" is not necessarily reached (e.g. user aborted).
        ["target duration", float, "Target measurement duration [min]"],
        ["target event count", fint, "Target event count for online gating"],
    ],
    # All setup-related keywords, except imaging
    "setup": [
        ["channel width", float, "Width of microfluidic channel [µm]"],
        ["chip identifier", lcstr, "Unique identifier of the chip used"],
        ["chip region", lcstr, "Imaged chip region (channel or reservoir)"],
        ["flow rate", float, "Flow rate in channel [µL/s]"],
        ["flow rate sample", float, "Sample flow rate [µL/s]"],
        ["flow rate sheath", float, "Sheath flow rate [µL/s]"],
        ["identifier", str, "Unique setup identifier"],
        # "medium" is one of CellCarrier, CellCarrierB, water, or other
        ["medium", str, "Medium used"],
        ["module composition", str, "Comma-separated list of modules used"],
        ["software version", str, "Acquisition software with version"],
        ["temperature", float, "Mean chip temperature [°C]"],
    ],
}

#: List of scalar (one scalar value per event) features. This
#: list does not include the `ml_score_???` features. If you
#: need find out whether a feature name is valid, please use
#: :func:`is_valid_feature`.
FEATURES_SCALAR = [
    ["area_cvx", "Convex area [px]"],
    # area_msd is the contour moment M00
    ["area_msd", "Measured area [px]"],
    ["area_ratio", "Porosity (convex to measured area ratio)"],
    # area_um is the convex area per definition
    ["area_um", "Area [µm²]"],
    ["aspect", "Aspect ratio of bounding box"],
    ["bright_avg", "Brightness average within contour [a.u.]"],
    ["bright_sd", "Brightness SD within contour [a.u.]"],
    ["circ", "Circularity"],
    # deform is computed from the convex contour
    ["deform", "Deformation"],
    ["emodulus", "Young's Modulus [kPa]"],
    # fl*_area, fl*_pos, and fl*_width values correspond to the
    # object for which the contour was found. For high concentrations,
    # these values could be error-prone due to the assignment from
    # false objects.
    ["fl1_area", "FL-1 area of peak [a.u.]"],
    # fl1_dist is set to zero if there is only one peak
    ["fl1_dist", "FL-1 distance between two first peaks [µs]"],
    ["fl1_max", "FL-1 maximum [a.u.]"],
    ["fl1_max_ctc", "FL-1 maximum, crosstalk-corrected [a.u.]"],
    ["fl1_npeaks", "FL-1 number of peaks"],
    ["fl1_pos", "FL-1 position of peak [µs]"],
    ["fl1_width", "FL-1 width [µs]"],
    ["fl2_area", "FL-2 area of peak [a.u.]"],
    ["fl2_dist", "FL-2 distance between two first peaks [µs]"],
    ["fl2_max", "FL-2 maximum [a.u.]"],
    ["fl2_max_ctc", "FL-2 maximum, crosstalk-corrected [a.u.]"],
    ["fl2_npeaks", "FL-2 number of peaks"],
    ["fl2_pos", "FL-2 position of peak [µs]"],
    ["fl2_width", "FL-2 width [µs]"],
    ["fl3_area", "FL-3 area of peak [a.u.]"],
    ["fl3_dist", "FL-3 distance between two first peaks [µs]"],
    ["fl3_max", "FL-3 maximum [a.u.]"],
    ["fl3_max_ctc", "FL-3 maximum, crosstalk-corrected [a.u.]"],
    ["fl3_npeaks", "FL-3 number of peaks"],
    ["fl3_pos", "FL-3 position of peak [µs]"],
    ["fl3_width", "FL-3 width [µs]"],
    ["frame", "Video frame number"],
    ["g_force", "Gravitational force in multiples of g"],
    # index starts with 1
    ["index", "Event index (Dataset)"],
    # index_online may have missing values (#71)
    ["index_online", "Event index (Online)"],
    # The inertia ratios of the event contours are defined by the
    # central second order moments of area (sqrt(m20/m02).
    ["inert_ratio_cvx", "Inertia ratio of convex contour"],
    ["inert_ratio_prnc", "Principal inertia ratio of raw contour"],
    ["inert_ratio_raw", "Inertia ratio of raw contour"],
    # This is an ancillary integer feature for visualizing the class
    # membership of individual events based on the `ml_score_???`
    # features.
    ["ml_class", "Most probable ML class"],
    ["nevents", "Total number of events in the same image"],
    ["pc1", "Principal component 1"],
    ["pc2", "Principal component 2"],
    # pos_x and pos_y are computed from the contour moments
    # "m10"/"m00" and "m01"/"m00" of the convex hull of "contour"
    ["pos_x", "Position along channel axis [µm]"],
    ["pos_y", "Position lateral in channel [µm]"],
    ["size_x", "Bounding box size x [µm]"],
    ["size_y", "Bounding box size y [µm]"],
    ["temp", "Chip temperature [°C]"],
    ["temp_amb", "Ambient temperature [°C]"],
    ["tilt", "Absolute tilt of raw contour"],
    ["time", "Event time [s]"],
    # Volume is computed from the raw contour (i.e. with exclusions).
    # Fun fact: If we had decided to compute it from the convex contour,
    # then we would have close to none pixelation effects ¯\_(ツ)_/¯.
    ["volume", "Volume [µm³]"],
]
# Add userdef features
for _i in range(10):
    FEATURES_SCALAR.append(["userdef{}".format(_i),
                            "User defined {}".format(_i)
                            ])

#: list of non-scalar features
FEATURES_NON_SCALAR = [
    # This is a (M, 2)-shaped array with integer contour coordinates
    ["contour", "Binary event contour image"],
    ["image", "Gray scale event image"],
    ["image_bg", "Gray scale event background image"],
    # This is the contour with holes filled
    ["mask", "Binary region labeling the event in the image"],
    # See FLUOR_TRACES for valid keys
    ["trace", "Dictionary of fluorescence traces"],
]

#: List of fluorescence traces
FLUOR_TRACES = [
    "fl1_median",
    "fl1_raw",
    "fl2_median",
    "fl2_raw",
    "fl3_median",
    "fl3_raw",
]


# CFG convenience lists and dicts
_cfg = copy.deepcopy(CFG_METADATA)
_cfg.update(CFG_ANALYSIS)
#: dict with section as keys and config parameter names as values
config_keys = {}
for _key in _cfg:
    config_keys[_key] = [it[0] for it in _cfg[_key]]
#: dict of dicts containing functions to convert input data
config_funcs = {}
for _key in _cfg:
    config_funcs[_key] = {}
    for _subkey, _type, __ in _cfg[_key]:
        config_funcs[_key][_subkey] = _type
#: dict of dicts containing the type of section parameters
config_types = {}
for _key in _cfg:
    config_types[_key] = {}
    for _subkey, _type, __ in _cfg[_key]:
        if _type in func_types:
            _type = func_types[_type]
        config_types[_key][_subkey] = _type
#: dict with metadata description
config_descr = {}
for _key in _cfg:
    config_descr[_key] = {}
    for _subkey, __, _descr in _cfg[_key]:
        config_descr[_key][_subkey] = _descr


# FEATURE convenience lists and dicts
#: list of feature names
feature_names = [_cc[0] for _cc in FEATURES_SCALAR + FEATURES_NON_SCALAR]
#: list of feature labels (same order as :const:`feature_names`
feature_labels = [_cc[1] for _cc in FEATURES_SCALAR + FEATURES_NON_SCALAR]
#: dict for converting feature names to labels
feature_name2label = {}
for _cc in FEATURES_SCALAR + FEATURES_NON_SCALAR:
    feature_name2label[_cc[0]] = _cc[1]

#: list of scalar feature names
scalar_feature_names = [_cc[0] for _cc in FEATURES_SCALAR]


def feature_exists(name, scalar_only=False):
    """Return True if `name` is a valid feature name

    This function not only checks whether `name` is in
    :const:`feature_names`, but also validates against
    the machine learning scores `ml_score_???` (where
    `?` can be a digit or a lower-case letter in the
    English alphabet).
    """
    valid = False
    if name in scalar_feature_names:
        # scalar feature
        valid = True
    elif not scalar_only and name in feature_names:
        # non-scalar feature
        valid = True
    else:
        # check whether we have an `ml_score_???` feature
        valid_chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        if (name.startswith("ml_score_")
            and len(name) == len("ml_score_???")
            and name[-3] in valid_chars
            and name[-2] in valid_chars
                and name[-1] in valid_chars):
            valid = True
    return valid


def get_feature_label(name, rtdc_ds=None):
    """Return the label corresponding to a feature name

    This function not only checks :const:`feature_name2label`,
    but also supports registered `ml_score_???` features.

    TODO: extract feature label from ancillary information
    when an rtdc_ds is given.
    """
    assert feature_exists(name)
    if name in feature_name2label:
        label = feature_name2label[name]
    else:
        # First check whether an ancillary feature with that
        # name exists.
        for af in AncillaryFeature.features:
            if af.feature_name == name:
                labelid = af.data.outputs.index(name)
                label = af.data.output_labels[labelid]
                break
        else:
            # If that did not work, use a generic name.
            label = "ML score {}".format(name[-3:].upper())
    return label


def scalar_feature_exists(name):
    """Convenience method wrapping `feature_exists(..., scalar_only=True)`"""
    return feature_exists(name, scalar_only=True)


def update_dfn_with_feature_info(feature, label, is_scalar):
    """Used by temporary features and plugin features to update the feature
    names and labels.
    """
    allowed_chars = "abcdefghijklmnopqrstuvwxyz_1234567890"
    _feat = "".join([f for f in feature if f in allowed_chars])
    if _feat != feature:
        raise ValueError("`feature` must only contain lower-case characters, "
                         "digits, and underscores; got '{}'!".format(feature))
    if label is None:
        label = "User defined feature {}".format(feature)
    if feature_exists(feature):
        raise ValueError("Feature '{}' already exists!".format(feature))

    # Populate the new feature in all dictionaries and lists
    # in `dclab.definitions`
    feature_names.append(feature)
    feature_labels.append(label)
    feature_name2label[feature] = label
    if is_scalar:
        scalar_feature_names.append(feature)
    return label


def remove_dfn_feature_info(feature, label):
    """Used by temporary features and plugin features to remove the feature
    names and labels from `dclab.definitions`.
    """
    feature_names.remove(feature)
    feature_labels.remove(label)
    feature_name2label.pop(feature)
    if feature in scalar_feature_names:
        scalar_feature_names.remove(feature)


def check_feature_shape(feature, data):
    data = np.array(data)
    # bug: reloaded contour is of the form: array([array([[1,2],[...]...])])
    # which means len(data.shape) == 1.
    if feature == "contour" and len(data.shape) == 1:
        pass
    elif len(data.shape) == 1 and not scalar_feature_exists(feature):
        raise ValueError("Feature '{}' is not a scalar feature, but a "
                         "1D array was given for `data`!".format(feature))
    elif len(data.shape) != 1 and scalar_feature_exists(feature):
        raise ValueError("Feature '{}' is a scalar feature, but the `data` "
                         "array is not 1D!".format(feature))
