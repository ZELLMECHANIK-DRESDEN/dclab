.. _sec_av_feat_temp:

===============================
User-defined temporary features
===============================
For specialized applications, the features defined internally in dclab might
not be enough to describe certain aspects of your data. For instance, let's say
you are interested in the mean overall fluorescence signal of each event
in channel 1 and you would like to filter the dataset according to that
information. You can define a temporary feature in your dataset without
modifying any files on disk.


Setting a temporary feature in a dataset
========================================
For this example, you can register the temporary feature `fl1_mean` and
manually set a corresponding filter for your dataset.

.. ipython::

    In [1]: import dclab

    In [2]: import numpy as np

    In [3]: ds = dclab.new_dataset("data/example_traces.rtdc")

    # register a temporary feature
    In [4]: dclab.register_temporary_feature(feature="fl1_mean")

    # compute the temporary feature
    In [5]: fl1_mean = np.array([np.mean(ds["trace"]["fl1_raw"][ii]) for ii in range(len(ds))])

    # set the temporary feature
    In [6]: dclab.set_temporary_feature(rtdc_ds=ds, feature="fl1_mean", data=fl1_mean)

    # do some filtering
    In [7]: ds.config["filtering"]["fl1_mean min"] = 4

    In [8]: ds.config["filtering"]["fl1_mean max"] = 200

    In [9]: ds.apply_filter()

    In [10]: print("Removed {} out of {} events!".format(np.sum(~ds.filter.all), len(ds)))


Accessing temporary features stored in data files
=================================================
It is also possible to store temporary features within datasets on disk.
At a later time point, you can then load this data file from disk with access
to those temporary features [1]_.

.. note::

    This is definitely not supported by Shape-Out, DCKit, or DCOR/DCOR-Aid.
    If you would like to compute additional features with Shape-Out, you
    should subscribe to the `issue about PluginFeature
    <https://github.com/ZELLMECHANIK-DRESDEN/dclab/issues/105>`_.

There are two ways of adding temporary features to an .rtdc data file.

- 1. With `h5py <https://docs.h5py.org>`_:

    .. code:: python

        import dclab
        import h5py
        import numpy as np

        # extract the feature data from the dataset
        with dclab.new_dataset("/path/to/data.rtdc") as ds:
            fl1_mean = np.array([np.mean(ds["trace"]["fl1_raw"][ii]) for ii in range(len(ds))])

        # write the feature to the HDF5 file
        with h5py.File("/path/to/data.rtdc", "a") as h5:
            h5["events"]["fl1_mean"] = fl1_mean

- 2. Via :func:`RTDCBase.export.hdf5 <dclab.rtdc_dataset.export.Export.hdf5>`:

    .. code:: python

        import dclab
        import h5py
        import numpy as np

        # register temporary feature
        dclab.register_temporary_feature(feature="fl1_mean")

        with dclab.new_dataset("/path/to/data.rtdc") as ds:
            # extract the feature information from the dataset
            fl1_mean = np.array([np.mean(ds["trace"]["fl1_raw"][ii]) for ii in range(len(ds))])
            # set the data
            dclab.set_temporary_feature(rtdc_ds=ds, feature="fl1_mean", data=fl1_mean)
            # export the data to a new file
            ds.export.hdf5("/path/to/data_with_fl1_mean.rtdc",
                           features=ds.features_innate + ["fl1_mean"])

If you wish to load the data at a later time point, you have to make sure
that you register the temporary feature before trying to access it.
This will not work::

    ds = dclab.new_dataset("/path/to/data_with_fl1_mean.rtdc")
    fl1_mean = ds["fl1_mean"]

But this works::

    dclab.register_temporary_feature(feature="fl1_mean")
    ds = dclab.new_dataset("/path/to/data_with_fl1_mean.rtdc")
    fl1_mean = ds["fl1_mean"]

And this works as well (registering after instantiation)::

    ds = dclab.new_dataset("/path/to/data_with_fl1_mean.rtdc")
    dclab.register_temporary_feature(feature="fl1_mean")
    fl1_mean = ds["fl1_mean"]


Please read the :ref:`code reference on temporary features
<cr_temp_feat>` for more
information.

.. [1] I know, storing *temporary* features on disk sounds like a
       counter-intuitive concept, but this is a very convenient extension
       of temporary features which came with almost no overhead.
       In a sense, it's still temporary, because you always have to register
       the feature before you can access it.