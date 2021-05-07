"""
.. versionadded:: 0.33.0
"""

from .. import definitions as dfn

from .fmt_hierarchy import RTDC_Hierarchy


_registered_temporary_features = []


def deregister_all():
    """Deregisters all temporary features"""
    for feat in list(_registered_temporary_features):
        deregister_temporary_feature(feat)


def deregister_temporary_feature(feature):
    """Convenience function for deregistering a temporary feature

    This method is mostly used during testing. It does not
    remove the actual feature data from any dataset; the data
    will stay in memory but is not accessible anymore through
    the public methods of the :class:`RTDCBase` user interface.
    """
    if feature in _registered_temporary_features:
        label = dfn.get_feature_label(feature)
        _registered_temporary_features.remove(feature)
        dfn.remove_dfn_feature_info(feature, label)


def register_temporary_feature(feature, label=None, is_scalar=True):
    """Register a new temporary feature

    Temporary features are custom features that can be defined ad hoc
    by the user. Temporary features are helpful when the integral
    features are not enough, e.g. for prototyping, testing, or
    collating with other data. Temporary features allow you to
    leverage the full functionality of :class:`RTDCBase` with
    your custom features (no need to go for a custom `pandas.Dataframe`).

    Parameters
    ----------
    feature: str
        Feature name; allowed characters are lower-case letters,
        digits, and underscores
    label: str
        Feature label used e.g. for plotting
    is_scalar: bool
        Whether or not the feature is a scalar feature
    """
    _ = dfn.update_dfn_with_feature_info(feature, label, is_scalar)
    _registered_temporary_features.append(feature)


def set_temporary_feature(rtdc_ds, feature, data):
    """Set temporary feature data for a dataset

    Parameters
    ----------
    rtdc_ds: dclab.RTDCBase
        Dataset for which to set the feature. Note that temporary
        features cannot be set for hierarchy children and that the
        length of the feature `data` must match the number of events
        in `rtdc_ds`.
    feature: str
        Feature name
    data: np.ndarray
        The data
    """
    if not dfn.feature_exists(feature):
        raise ValueError(
            "Temporary feature '{}' has not been registered!".format(feature))
    if isinstance(rtdc_ds, RTDC_Hierarchy):
        raise NotImplementedError("Setting temporary features for hierarchy "
                                  "children not implemented yet!")
    if len(data) != len(rtdc_ds):
        raise ValueError("The temporary feature `data` must have same length "
                         "as the dataset. Expected length {}, got length "
                         "{}!".format(len(rtdc_ds), len(data)))
    dfn.check_feature_shape(feature, data)
    rtdc_ds._usertemp[feature] = data
