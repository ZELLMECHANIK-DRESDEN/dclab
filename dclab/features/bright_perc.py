"""
Computation of the 10th and 90th percentile of grayscale values inside the
RT-DC event image mask with background-correction taken into account.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def get_bright_perc(mask: npt.NDArray[bool] | list[npt.NDArray[bool]],
                    image: npt.NDArray | list[npt.NDArray],
                    image_bg: npt.NDArray | list[npt.NDArray],
                    bg_off: float | npt.NDArray = None
                    ) -> (tuple[float, float] |
                          tuple[npt.NDArray, npt.NDArray]):
    """Compute 10th and 90th percentile of the bg-corrected event brightness

    The background-corrected event brightness is defined by the
    gray-scale values of the background-corrected image data
    within the event mask area.

    Parameters
    ----------
    mask: ndarray or list of ndarrays of shape (M,N) and dtype bool
        The mask values, True where the event is located in `image`.
    image: ndarray or list of ndarrays of shape (M,N)
        A 2D array that holds the image in form of grayscale values
        of an event.
    image_bg: ndarray or list of ndarrays of shape (M,N)
        A 2D array that holds the background image for the same event.
    bg_off: float or 1D ndarray
        Additional offset value that is added to `image_bg` before
        background correction

    Returns
    -------
    bright_perc_10: float or ndarray of size N
        10th percentile of brightness
    bright_perc_90: float or ndarray of size N
        90th percentile of brightness
    """
    if isinstance(mask, np.ndarray) and len(mask.shape) == 2:
        # We have a single image
        image_bg = [image_bg]
        image = [image]
        mask = [mask]
        ret_list = False
    else:
        ret_list = True

    length = min(len(mask), len(image), len(image_bg))

    p10 = np.zeros(length, dtype=np.float64) * np.nan
    p90 = np.zeros(length, dtype=np.float64) * np.nan

    for ii in range(length):
        # cast to integer before subtraction
        imgi = np.array(image[ii], dtype=int) - image_bg[ii]
        mski = mask[ii]
        # Assign results
        p10[ii], p90[ii] = np.percentile(imgi[mski], q=[10, 90])

    if bg_off:
        p10 -= bg_off
        p90 -= bg_off

    if ret_list:
        return p10, p90
    else:
        return p10[0], p90[0]
