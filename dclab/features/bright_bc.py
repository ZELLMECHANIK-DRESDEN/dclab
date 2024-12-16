"""
Computation of mean and standard deviation of grayscale values inside the
RT-DC event image mask with background-correction taken into account.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def get_bright_bc(mask: npt.NDArray[bool] | list[npt.NDArray[bool]],
                  image: npt.NDArray | list[npt.NDArray],
                  image_bg: npt.NDArray | list[npt.NDArray],
                  bg_off: float | npt.NDArray = None,
                  ret_data: str = "avg,sd"
                  ) -> (float |
                        npt.NDArray |
                        tuple[float, float] |
                        tuple[npt.NDArray, npt.NDArray]):
    """Compute avg and/or std of the background-corrected event brightness

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
    ret_data
        A comma-separated list of metrices to compute
        - "avg": compute the average
        - "sd": compute the standard deviation
        Selected metrics are returned in alphabetical order.

    Returns
    -------
    bright_avg: float or ndarray of size N
        Average image data within the contour
    bright_std: float or ndarray of size N
        Standard deviation of image data within the contour
    """
    # This method is based on a pull request by Maik Herbig.
    ret_avg = "avg" in ret_data
    ret_std = "sd" in ret_data

    if ret_avg + ret_std == 0:
        raise ValueError("No valid metrices selected!")

    if isinstance(mask, np.ndarray) and len(mask.shape) == 2:
        # We have a single image
        image_bg = [image_bg]
        image = [image]
        mask = [mask]
        if bg_off is not None:
            bg_off = np.atleast_1d(bg_off)
        ret_list = False
    else:
        ret_list = True

    length = min(len(mask), len(image), len(image_bg))

    # Results are stored in a separate array initialized with nans
    if ret_avg:
        avg = np.zeros(length, dtype=np.float64) * np.nan
    if ret_std:
        std = np.zeros(length, dtype=np.float64) * np.nan

    for ii in range(length):
        # cast to integer before subtraction
        imgi = np.array(image[ii], dtype=int) - image_bg[ii]
        mski = mask[ii]
        # Assign results
        if ret_avg:
            avg[ii] = np.mean(imgi[mski])
        if ret_std:
            std[ii] = np.std(imgi[mski])

    results = []
    # Keep alphabetical order
    if ret_avg:
        if bg_off is not None:
            avg -= bg_off
        results.append(avg)
    if ret_std:
        results.append(std)

    if not ret_list:
        # Only return scalars
        results = [r[0] for r in results]

    if ret_avg + ret_std == 1:
        # Only return one column
        return results[0]

    return results
