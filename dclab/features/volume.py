"""Volume computation based on contour revolution"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def get_volume(
        cont: npt.NDArray | list[npt.NDArray],
        pos_x: float | npt.NDArray,
        pos_y: float | npt.NDArray,
        pix: float,
        fix_orientation: bool = False) -> float | npt.NDArray:
    """Calculate the volume of a polygon revolved around an axis

    The volume estimation assumes rotational symmetry.

    Parameters
    ----------
    cont: ndarray or list of ndarrays of shape (N,2)
        A 2D array that holds the contour of an event [px]
        e.g. obtained using `mm.contour` where  `mm` is an instance
        of `RTDCBase`. The first and second columns of `cont`
        correspond to the x- and y-coordinates of the contour.
    pos_x: float or ndarray of length N
        The x coordinate(s) of the centroid of the event(s) [µm]
        e.g. obtained using `mm.pos_x`
    pos_y: float or ndarray of length N
        The y coordinate(s) of the centroid of the event(s) [µm]
        e.g. obtained using `mm.pos_y`
    pix
        The detector pixel size in µm.
        e.g. obtained using: `mm.config["imaging"]["pixel size"]`
    fix_orientation
        If set to True, make sure that the orientation of the
        contour is counter-clockwise in the r-z plane
        (see :func:`vol_revolve`). This is False by default, because
        (1) Shape-In always stores the contours in the correct
        orientation and (2) there may be events with high porosity
        where "fixing" the orientation makes things worse and a
        negative volume is returned.

    Returns
    -------
    volume
        volume in um^3

    Notes
    -----
    The computation of the volume is based on a full rotation of the
    upper and the lower halves of the contour from which the
    average is then used.

    The volume is computed radially from the center position
    given by (`pos_x`, `pos_y`). For sufficiently smooth contours,
    such as densely sampled ellipses, the center position does not
    play an important role. For contours that are given on a coarse
    grid, as is the case for RT-DC, the center position must be
    given.

    References
    ----------
    - https://de.wikipedia.org/wiki/Kegelstumpf#Formeln
    - Yields identical results to the Matlab script by Geoff Olynyk
      <https://de.mathworks.com/matlabcentral/fileexchange/36525-volrevolve>`_
    """
    if np.isscalar(pos_x):
        cont = [cont]
        ret_list = False
    else:
        ret_list = True

    # Convert input to 1D arrays
    pos_x = np.atleast_1d(pos_x)
    pos_y = np.atleast_1d(pos_y)

    if pos_x.size != pos_y.size:
        raise ValueError("Size of `pos_x` and `pos_y` must match!")

    if pos_x.size > 1 and len(cont) <= 1:
        raise ValueError("Number of given contours too small!")

    # results are stored in a separate array initialized with nans
    v_avg = np.zeros_like(pos_x, dtype=np.float64) * np.nan

    # v_avg has the shape of `pos_x`. We are iterating over the smallest
    # length for `cont` and `pos_x`.
    for ii in range(min(len(cont), pos_x.shape[0])):
        # If the contour has less than 4 pixels, the computation will fail.
        # In that case, the value np.nan is already assigned.
        cc = cont[ii]
        if cc.shape[0] >= 4:
            # Center contour coordinates with given centroid
            contour_x = cc[:, 0] - pos_x[ii] / pix
            contour_y = cc[:, 1] - pos_y[ii] / pix
            # Switch to r and z to follow notation of vol_revolve
            # (In RT-DC the axis of rotation is x, but for vol_revolve
            # we need the axis vertically)
            contour_r = contour_y
            contour_z = contour_x
            if fix_orientation:
                # Make sure the contour is counter-clockwise
                contour_r, contour_z = counter_clockwise(contour_r, contour_z)

            # Compute right volume
            # Which points are at negative r-values (r<0)?
            inx_neg = np.where(contour_r < 0)
            # These points will be shifted up to r=0 directly on the z-axis
            contour_right = np.copy(contour_r)
            contour_right[inx_neg] = 0
            vol_right = vol_revolve(contour_right, contour_x, pix)

            # Compute left volume
            # Which points are at positive r-values? (r>0)?
            idx_pos = np.where(contour_r > 0)
            # These points will be shifted down to y=0 to build an x-axis
            contour_left = np.copy(contour_r)
            contour_left[idx_pos] = 0
            # Now we still have negative r values, but vol_revolve needs
            # positive values, so we flip the sign...
            contour_left[:] *= -1
            # ... but in doing so, we have switched to clockwise rotation
            # and we need to pass the array in reverse order
            vol_left = vol_revolve(contour_left[::-1], contour_x[::-1], pix)

            # Compute the average
            v_avg[ii] = (vol_right + vol_left) / 2

    if not ret_list:
        # Do not return a list if the input contour was not in a list
        v_avg = v_avg[0]

    return v_avg


def counter_clockwise(cx: npt.NDArray, cy: npt.NDArray) -> tuple[float, float]:
    """Put contour coordinates into counter-clockwise order

    Parameters
    ----------
    cx, cy
        The x- and y-coordinates of the contour

    Returns
    -------
    cx_cc, cy_cc
        The x- and y-coordinates of the contour in
        counter-clockwise orientation.

    Notes
    -----
    The contour must be centered around (0, 0).
    """
    # test orientation
    angles = np.unwrap(np.arctan2(cy, cx))
    grad = np.diff(angles)
    if np.average(grad) < 0:
        return cx[::-1], cy[::-1]
    else:
        return cx, cy


def vol_revolve(r: npt.NDArray,
                z: npt.NDArray,
                point_scale: float = 1.) -> float | npt.NDArray:
    r"""Calculate the volume of a polygon revolved around the Z-axis

    This implementation yields the same results as the volRevolve
    Matlab function by Geoff Olynyk (from 2012-05-03)
    https://de.mathworks.com/matlabcentral/fileexchange/36525-volrevolve.

    The difference here is that the volume is computed using (a much
    more approachable) implementation using the volume of a truncated
    cone (https://de.wikipedia.org/wiki/Kegelstumpf).

    .. math::

      V = \frac{h \cdot \pi}{3} \cdot (R^2 + R \cdot r + r^2)

    Where :math:`h` is the height of the cone and :math:`r` and
    `R` are the smaller and larger radii of the truncated cone.

    Each line segment of the contour resembles one truncated cone. If
    the z-step is positive (counter-clockwise contour), then the
    truncated cone volume is added to the total volume. If the z-step
    is negative (e.g. inclusion), then the truncated cone volume is
    removed from the total volume.

    .. versionchanged:: 0.37.0

      The volume in previous versions was overestimated by on average
      2µm³.

    Parameters
    ----------
    r
        radial coordinates (perpendicular to the z axis)
    z
        coordinate along the axis of rotation
    point_scale
        point size in your preferred units; The volume is multiplied
        by a factor of `point_scale**3`.

    Notes
    -----
    The coordinates must be given in counter-clockwise order,
    otherwise the volume will be negative.
    """
    r = np.array(r).flatten()
    z = np.array(z).flatten()

    # sanity checks
    assert len(r) == len(z)
    assert len(r) >= 3
    assert len(r.shape) == len(z.shape) == 1
    assert np.all(r >= 0)

    # make sure we have a closed contour
    if (r[-1] != r[0]) or (z[-1] != z[0]):
        # We have an open contour - close it.
        r = np.resize(r, len(r) + 1)
        z = np.resize(z, len(z) + 1)

    rp = r[:-1]

    # array of radii differences: R - r
    dr = np.diff(r)
    # array of height differences: h
    dz = np.diff(z)

    # If we expand the function in the doc string with
    # dr = R - r and dz = h, then we get three terms for the volume
    # (as opposed to four terms in Olynyk's script). Those three terms
    # all resemble area slices multiplied by the z-distance dz.
    a1 = 3 * rp ** 2
    a2 = 3 * rp * dr
    a3 = dr ** 2

    # Note that the formula for computing the volume is symmetric
    # with respect to r and R. This means that it does not matter
    # which sign dr has (R and r are always positive). Since this
    # algorithm assumes that the contour is ordered counter-clockwise,
    # positive dz means adding to the contour while negative dz means
    # subtracting from the contour (see test functions for more details).
    # Conveniently so, dz only appears one time in this formula, so
    # we can take the sign of dz as it is (Otherwise, we would have
    # to take the absolute value of every truncated cone volume and
    # multiply it by np.sign(dz)).
    v = np.pi / 3 * dz * np.abs(a1 + a2 + a3)
    vol = np.sum(v) * point_scale ** 3

    return vol
