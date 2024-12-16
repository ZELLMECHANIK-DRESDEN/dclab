.. _sec_av_dc_io:

===========
DC data I/O
===========

When working with DC data, you will inevitably run into the situation where
you would like to write some part of dataset (be it an .rtdc file or data
on DCOR) to a new file. Depending on the situation, one or more of the
following subsection will probably cover what you need.


.. _sec_av_dc_io_export:

Exporting data
==============
The :class:`RTDCBase <.RTDCBase>` class has the attribute
:attr:`RTDCBase.export <.RTDCBase.export>` which allows to export event
data to several data file formats. See :ref:`sec_ref_rtdc_export` for more
information.

.. ipython::

    In [1]: import dclab

    In [2]: ds = dclab.new_dataset("data/example.rtdc")

    # Restrict the deformation to 0.15
    In [3]: ds.config["filtering"]["deform min"] = 0

    In [4]: ds.config["filtering"]["deform max"] = .15

    In [5]: ds.apply_filter()

    In [6]: print("Deformation mean before filtering:", ds["deform"][:].mean())

    In [7]: print("Deformation mean after filtering:", ds["deform"][ds.filter.all].mean())

    # Export to .tsv
    In [8]: ds.export.tsv(path="export_example.tsv",
       ...:               features=["area_um", "deform"],
       ...:               filtered=True,
       ...:               override=True)
       ...:

    # Export to .rtdc
    In [9]: ds.export.hdf5(path="export_example.rtdc",
       ...:                features=["area_um", "aspect", "deform"],
       ...:                filtered=True,
       ...:                override=True)
       ...:

Note that data exported as HDF5 files can be loaded with dclab
(reproducing the previously computed statistics - without filters).

.. ipython::

    In [10]: ds2 = dclab.new_dataset("export_example.rtdc")

    In [11]: ds2["deform"][:].mean()


.. _sec_av_dc_io_write:

Writing to an .rtdc file
========================

If you would like to create your own .rtdc files, you can
make use of the :class:`RTDCWriter <dclab.rtdc_dataset.writer.RTDCWriter>` class.

.. ipython::

    In [4]: with dclab.RTDCWriter("my-data.rtdc", mode="reset") as hw:
       ...:     hw.store_metadata({"experiment": {"sample": "my sample",
       ...:                                       "run index": 1}})
       ...:     hw.store_feature("deform", np.random.rand(100))
       ...:     hw.store_feature("area_um", np.random.rand(100))

    In [5]: ds_custom = dclab.new_dataset("my-data.rtdc")

    In [6]: print(ds_custom.features)

    In [7]: print(ds_custom.config["experiment"])


RTDCWriter feature writing:
###########################

1. Simple feature writing (append mode):

.. ipython::

    In [4]: with dclab.RTDCWriter("my-data.rtdc", mode="append") as hw:
       ...:     hw.store_feature("deform", np.random.rand(100))

    In [5]: ds_custom = dclab.new_dataset("my-data.rtdc")

    In [6]: print(ds_custom.features)


2. Advanced feature writing (replace mode):

.. ipython::

    In [4]: with dclab.RTDCWriter("my-data.rtdc", mode="replace") as hw:
       ...:     hw.store_feature("deform", np.random.rand(100))

    In [5]: ds_custom = dclab.new_dataset("my-data.rtdc")

    In [6]: print(ds_custom.features)

.. _sec_av_dc_io_copy:

Copying (parts of) a dataset
============================

In some situations, you would only like to copy an entire feature column
from one dataset to a new file without modification. The :mod:`copier
<dclab.rtdc_dataset.copier>` submodule enables this on a low-level.

- Use the :func:`.rtdc_copy` method to create a compressed version of a DC
  dataset opened as an HDF5 file (:class:`.RTDC_HDF5` or :class:`.RTDC_S3`).
- Use the :func:`.h5ds_copy` method to copy parts of an HDF5 dataset to
  another HDF5 file, with the option to enforce compression (if the source
  :class:`h5py.Dataset` is not compressed properly already).
