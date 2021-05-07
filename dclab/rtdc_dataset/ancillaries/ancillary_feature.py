"""Computation of ancillary features

Ancillary features are computed on-the-fly in dclab if the
required data are available. The features are registered here
and are computed when `RTDCBase.__getitem__` is called with
the respective feature name. When `RTDCBase.__contains__` is
called with the feature name, then the feature is not yet
computed, but the prerequisites are evaluated:

.. ipython::
    :okwarning:

    In [1]: import dclab

    In [2]: ds = dclab.new_dataset("data/example.rtdc")

    In [4]: ds.config["calculation"]["emodulus lut"] = "LE-2D-FEM-19"

    In [3]: ds.config["calculation"]["emodulus medium"] = "CellCarrier"

    In [5]: ds.config["calculation"]["emodulus temperature"] = 23.0

    In [6]: "emodulus" in ds  # nothing is computed

    In [7]: ds["emodulus"] # now data is computed and cached

Once the data has been computed, `RTDCBase` caches it in
the `_ancillaries` property dict together with a hash
that is computed with `AncillaryFeature.hash`. The hash
is computed from the feature data `req_features` and the
configuration metadata `req_config`.
"""

import hashlib
import warnings

import numpy as np

from ...util import obj2bytes
from ... import definitions as dfn


class BadFeatureSizeWarning(UserWarning):
    pass


class AncillaryFeature:
    #: All ancillary features registered
    features = []
    #: All feature names registered
    feature_names = []

    def __init__(self, feature_name, method, req_config=[], req_features=[],
                 req_func=lambda x: True, priority=0, data=None):
        """A data feature that is computed from existing data

        Parameters
        ----------
        feature_name: str
            The name of the ancillary feature, e.g. "emodulus".
        method: callable
            The method that computes the feature. This method
            takes an instance of `RTDCBase` as argument.
        req_config: list
            Required configuration parameters to compute the feature,
            e.g. ["calculation", ["emodulus lut", "emodulus viscosity"]]
        req_features: list
            Required existing features in the dataset,
            e.g. ["area_cvx", "deform"]
        req_func: callable
            A function that takes an instance of `RTDCBase` as an
            argument and checks whether any other necessary criteria
            are met. By default, this is a lambda function that returns
            True. The function should return False if the necessary
            criteria are not met. This function may also return a
            hashable object (via :func:`dclab.util.objstr`) instead of
            True, if the criteria are subject to change. In this case,
            the return value is used for identifying the cached
            ancillary feature.

            .. versionchanged:: 0.27.0
                Support non-boolean return values for caching purposes.

        priority: int
            The priority of the feature; if there are multiple
            AncillaryFeature defined for the same feature_name,
            then the priority of the features defines which feature
            returns True in `self.is_available`. A higher value
            means a higher priority.
        data: object
            Any other data relevant for the feature (e.g. the ML
            model for computing 'ml_score_xxx' features)

        Notes
        -----
        `req_config` and `req_features` are used to test whether the
        feature can be computed in `self.is_available`.
        """
        self.feature_name = feature_name
        self.method = method
        self.req_config = req_config
        self.req_features = req_features
        self.req_func = req_func
        self.priority = priority
        self.data = data

        # register this feature
        AncillaryFeature.features.append(self)
        AncillaryFeature.feature_names.append(feature_name)

    def __repr__(self):
        repre = "<{} '{}' (priority {}) at {}>".format(
            self.__class__.__name__,
            self.feature_name,
            self.priority,
            hex(id(self)))
        return repre

    @staticmethod
    def available_features(rtdc_ds):
        """Determine available features for an RT-DC dataset

        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The dataset to check availability for

        Returns
        -------
        features: dict
            Dictionary with feature names as keys and instances
            of `AncillaryFeature` as values.
        """
        cols = {}
        for inst in AncillaryFeature.features:
            if inst.is_available(rtdc_ds):
                cols[inst.feature_name] = inst
        return cols

    @staticmethod
    def get_instances(feature_name):
        """Return all instances that compute `feature_name`"""
        feats = []
        for ft in AncillaryFeature.features:
            if ft.feature_name == feature_name:
                feats.append(ft)
        return feats

    @staticmethod
    def check_data_size(rtdc_ds, data_dict):
        """Check the feature data is the correct size. If it isn't, resize it.

        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The dataset from which the features are computed
        data_dict: dict
            Dictionary with `AncillaryFeature.feature_name` as keys and the
            computed data features (to be resized) as values.

        Returns
        -------
        data_dict: dict
            Dictionary with `feature_name` as keys and the correctly resized
            data features as values.
        """
        for key in data_dict:
            dsize = len(rtdc_ds) - len(data_dict[key])
            if dsize > 0:
                msg = "Growing feature {} in {} by {} to match event number!"
                warnings.warn(msg.format(key, rtdc_ds, abs(dsize)),
                              BadFeatureSizeWarning)
                data_dict[key].resize(len(rtdc_ds), refcheck=False)
                data_dict[key][-dsize:] = np.nan
            elif dsize < 0:
                msg = "Shrinking feature {} in {} by {} to match event number!"
                warnings.warn(msg.format(key, rtdc_ds, abs(dsize)),
                              BadFeatureSizeWarning)
                data_dict[key].resize(len(rtdc_ds), refcheck=False)
            if isinstance(data_dict[key], np.ndarray):
                data_dict[key].setflags(write=False)
            elif isinstance(data_dict[key], list):
                for item in data_dict[key]:
                    if isinstance(item, np.ndarray):
                        item.setflags(write=False)
        return data_dict

    def compute(self, rtdc_ds):
        """Compute the feature with self.method. All ancillary features that
        share the same method will also be populated automatically.

        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The dataset to compute the feature for

        Returns
        -------
        data_dict: dict
            Dictionary with `AncillaryFeature.feature_name` as keys and the
            computed data features (read-only) as values.
        """
        data_dict = self.method(rtdc_ds)
        if not isinstance(data_dict, dict):
            data_dict = {self.feature_name: data_dict}
        data_dict = AncillaryFeature.check_data_size(rtdc_ds, data_dict)
        for key in data_dict:
            dfn.check_feature_shape(self.feature_name, data_dict[key])
        return data_dict

    def hash(self, rtdc_ds):
        """Used for identifying an ancillary computation

        The data columns and the used configuration keys/values
        are hashed.
        """
        hasher = hashlib.md5()
        # data columns
        for col in self.req_features:
            hasher.update(obj2bytes(rtdc_ds[col]))
        # config keys
        for sec, keys in self.req_config:
            for key in keys:
                val = rtdc_ds.config[sec][key]
                data = "{}:{}={}".format(sec, key, val)
                hasher.update(obj2bytes(data))
        # custom requirement function hash
        reqret = self.req_func(rtdc_ds)
        if not isinstance(reqret, bool):
            # add to hash if not a boolean
            hasher.update(obj2bytes(reqret))
        return hasher.hexdigest()

    def is_available(self, rtdc_ds, verbose=False):
        """Check whether the feature is available

        Parameters
        ----------
        rtdc_ds: instance of RTDCBase
            The dataset to check availability for

        Returns
        -------
        available: bool
            `True`, if feature can be computed with `compute`

        Notes
        -----
        This method returns `False` for a feature if there
        is a feature defined with the same name but with
        higher priority (even if the feature would be
        available otherwise).
        """
        # Check config keys
        for item in self.req_config:
            section, keys = item
            if section not in rtdc_ds.config:
                if verbose:
                    print("{} not in config".format(section))
                return False
            else:
                for key in keys:
                    if key not in rtdc_ds.config[section]:
                        if verbose:
                            print("{} not in config['{}']".format(key,
                                                                  section))
                        return False
        # Check features
        for col in self.req_features:
            if col not in rtdc_ds:
                return False
        # Check priorities of other features
        for of in AncillaryFeature.features:
            if of == self:
                # nothing to compare
                continue
            elif of.feature_name == self.feature_name:
                # same feature name
                if of.priority <= self.priority:
                    # lower priority, ignore
                    continue
                else:
                    # higher priority
                    if of.is_available(rtdc_ds):
                        # higher priority is available, thus
                        # this feature is not available
                        return False
                    else:
                        # higher priority not available
                        continue
            else:
                # other feature
                continue
        # Check user-defined function
        if not self.req_func(rtdc_ds):
            return False
        return True
