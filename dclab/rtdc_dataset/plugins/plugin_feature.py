"""
.. versionadded:: 0.34.0
"""

import pathlib
import importlib
import sys

from ... import definitions as dfn
from ..ancillaries import AncillaryFeature


class PluginImportError(BaseException):
    pass


def load_plugin_feature(plugin_path):
    """find an instanciate a PlugInFeature from a user-defined script"""
    info = import_plugin_feature_script(plugin_path)
    plugin_list = []
    for ii in range(len(info["feature names"])):
        ancill_info = {
            "feature_name": info["feature names"][ii],
            "method": info["method"],
            "req_config": info["config required"],
            "req_features": info["features required"],
            "req_func": info["method check required"],
            "priority": info["priority"],
        }
        feature_label = info["feature labels"][ii]
        is_scalar = info["scalar feature"]
        plugin_list.append(PlugInFeature(
            feature_label, is_scalar, plugin_path, info, **ancill_info))
        # add feature label etc
    return plugin_list


def import_plugin_feature_script(plugin_path):
    # find script, return info dict
    path = pathlib.Path(plugin_path)
    try:
        # insert the plugin directory to sys.path so we can import it
        sys.path.insert(-1, str(path.parent))
        plugin = importlib.import_module(path.stem)
    except ModuleNotFoundError:
        raise PluginImportError("The plugin could be not be found at "
                                f"'{plugin_path}'")
    finally:
        # undo our path insertion
        sys.path.pop(0)

    return plugin.info


def remove_plugin_feature(plugin_instance):
    """Convenience function for removing a plugin instance"""
    if isinstance(plugin_instance, PlugInFeature):
        if plugin_instance.feature_name in PlugInFeature.feature_names:
            PlugInFeature.feature_names.remove(plugin_instance.feature_name)
            dfn.remove_dfn_feature_info(
                plugin_instance.feature_name,
                plugin_instance.feature_label)
        PlugInFeature.features.remove(plugin_instance)
    else:
        raise TypeError(f"Type {type(plugin_instance)} should be an instance "
                        "of PlugInFeature.")


def remove_all_plugin_features():
    """Convenience function for removing all plugin instances"""
    for plugin_instance in reversed(PlugInFeature.features):
        if isinstance(plugin_instance, PlugInFeature):
            remove_plugin_feature(plugin_instance)


class PlugInFeature(AncillaryFeature):
    def __init__(self, feature_label, is_scalar,
                 plugin_path, info, **kwargs):
        """Child class of `AncillaryFeature` which allows a user to create
        their own features. See the dclab repo examples/plugins folder for
        example plugins.
        """
        self.plugin_path = plugin_path
        self.plugin_info = info
        self.feature_label = feature_label
        self.is_scalar = is_scalar
        self._update_feature_name_and_label(
            feature_name=kwargs["feature_name"])
        super().__init__(**kwargs)

    def _update_feature_name_and_label(self, feature_name):
        """Wrapper on the `dclab.dfn.update_dfn_with_feature_info` function for
        handling cases when `feature_label=""`

        This should be excecuted before initializing the super class
        (AncillaryFeature). If we don't do this, then `remove_plugin_feature`
        may end up removing innate features e.g., "deform".

        """
        if self.feature_label == "":
            self.feature_label = None
        self.feature_label = dfn.update_dfn_with_feature_info(
            feature_name, self.feature_label, self.is_scalar)
