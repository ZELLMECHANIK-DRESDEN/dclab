"""Computation of event contour from event mask"""
from __future__ import annotations

from collections import deque
import numbers

import numpy as np
import numpy.typing as npt

# equivalent to
# from skimage.measure import find_contours
from ..external.skimage.measure import find_contours


class NoValidContourFoundError(BaseException):
    pass


class LazyContourList(object):
    def __init__(self, masks: npt.ArrayLike, max_events: int = 1000) -> None:
        """A list-like object that computes contours upon indexing

        Parameters
        ----------
        masks
            3D array of masks, may be an HDF5 dataset or any other
            structure that supports indexing
        max_events
            maximum number of contours to keep in the contour list;
            set to 0/False/None to cache all contours

        .. versionchanged:: 0.58.3

            Added the `max_events` parameter which now makes this class
            a lazy, least-recently-used contour list. To achieve the
            old behavior (which may fill up your memory), set
            `max_events=None`.
        """
        self.masks = masks
        self.contours = deque(maxlen=max_events or None)
        self.indices = deque(maxlen=max_events or None)
        #: used for hashing in ancillary features
        self.identifier = str(masks[0][:].tobytes())
        self.shape = len(masks), np.nan, 2

    def __getitem__(self, idx) -> npt.NDArray:
        """Compute contour(s) if not already in self.contours"""
        if not isinstance(idx, numbers.Integral):
            # slicing!
            indices = np.arange(len(self))[idx]
            output = []
            # populate the output list
            for evid in indices:
                output.append(self.__getitem__(evid))
            return output
        else:
            try:
                # Is the contour already available?
                idx_q = self.indices.index(idx)
            except ValueError:
                # The contour is not there. Compute it.
                try:
                    cont = get_contour(self.masks[idx])
                except BaseException as e:
                    e.args = (f"Event {idx}, {e.args[0]}",)
                    raise
            else:
                # Get the contour from deque
                cont = self.contours[idx_q]
            # If we got here, it means that we have computed a contour
            # successfully. Store it in the deque.
            self.contours.append(cont)
            self.indices.append(idx)
            return cont

    def __len__(self):
        return len(self.masks)


def get_contour(mask: npt.NDArray[bool]) -> npt.NDArray | list[npt.NDArray]:
    """Compute the image contour from a mask

    The contour is computed in a very inefficient way using scikit-image
    and a conversion of float coordinates to pixel coordinates.

    Parameters
    ----------
    mask: binary ndarray of shape (M,N) or (K,M,N)
        The mask outlining the pixel positions of the event.
        If a 3d array is given, then `K` indexes the individual
        contours.

    Returns
    -------
    cont: ndarray or list of K ndarrays of shape (J,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.
    """
    if isinstance(mask, np.ndarray) and len(mask.shape) == 2:
        mask = [mask]
        ret_list = False
    else:
        ret_list = True
    contours = []

    for mi in mask:
        # This is only 10% slower than doing:
        # conts, _ = cv2.findContours(np.array(mi, dtype=np.uint8),
        #                             cv2.RETR_EXTERNAL,
        #                             cv2.CHAIN_APPROX_NONE)
        # c2 = conts[0].reshape(-1, 2)
        conts = find_contours(mi.transpose(),
                              level=.9999,
                              positive_orientation="low",
                              fully_connected="high")
        # get the longest contour
        c0 = sorted(conts, key=lambda x: len(x))[-1]
        # round all coordinates to pixel values
        c1 = np.asarray(np.round(c0), int)
        # remove duplicates
        c2 = remove_duplicates(c1)
        if len(c2) == 0:
            raise NoValidContourFoundError("No contour found!")
        contours.append(c2)
    if ret_list:
        return contours
    else:
        return contours[0]


def get_contour_lazily(mask: npt.NDArray[bool]) -> \
        npt.NDArray | LazyContourList:
    """Like :func:`get_contour`, but computes contours on demand

    Parameters
    ----------
    mask: binary ndarray of shape (M,N) or (K,M,N)
        The mask outlining the pixel positions of the event.
        If a 3d array is given, then `K` indexes the individual
        contours.

    Returns
    -------
    cont: ndarray or LazyContourList of K ndarrays of shape (J,2)
        A 2D array that holds the contour of an event (in pixels)
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.
    """
    if isinstance(mask, np.ndarray) and len(mask.shape) == 2:
        # same behavior as `get_contour`
        cont = get_contour(mask=mask)
    else:
        cont = LazyContourList(masks=mask)
    return cont


def remove_duplicates(cont: npt.NDArray) -> npt.NDArray:
    """Remove duplicates in a circular contour"""
    x = np.resize(cont, (len(cont) + 1, 2))
    selection = np.ones(len(x), dtype=bool)
    selection[1:] = ~np.prod((x[1:] == x[:-1]), axis=1, dtype=bool)
    return x[selection][:-1]
