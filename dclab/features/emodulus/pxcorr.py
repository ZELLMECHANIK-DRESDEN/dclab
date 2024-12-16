"""Pixelation correction definitions"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def corr_deform_with_area_um(area_um: float | npt.NDArray,
                             px_um: float = 0.34) -> float | npt.NDArray:
    """Deformation correction for area_um-deform data

    The contour in RT-DC measurements is computed on a
    pixelated grid. Due to sampling problems, the measured
    deformation is overestimated and must be corrected.

    The correction formula is described in :cite:`Herold2017`.

    Parameters
    ----------
    area_um
        Apparent (2D image) area in µm² of the event(s)
    px_um
        The detector pixel size in µm.

    Returns
    -------
    deform_delta
        Error of the deformation of the event(s) that must be
        subtracted from `deform`.
        deform_corr = deform -  deform_delta
    """
    # A triple-exponential decay can be used to correct for pixelation
    # for apparent cell areas between 10 and 1250µm².
    # For 99 different radii between 0.4 μm and 20 μm circular objects were
    # simulated on a pixel grid with the pixel resolution of 340 nm/pix. At
    # each radius 1000 random starting points were created and the
    # obtained contours were analyzed in the same fashion as RT-DC data.
    # A convex hull on the contour was used to calculate the size (as area)
    # and the deformation.
    # The pixel size correction `pxscale` takes into account the pixel size
    # in the pixelation correction formula.
    pxscale = (.34 / px_um)**2
    offs = 0.0012
    exp1 = 0.020 * np.exp(-area_um * pxscale / 7.1)
    exp2 = 0.010 * np.exp(-area_um * pxscale / 38.6)
    exp3 = 0.005 * np.exp(-area_um * pxscale / 296)
    delta = offs + exp1 + exp2 + exp3

    return delta


def corr_deform_with_volume(volume: float | npt.NDArray,
                            px_um: float = 0.34) -> float | npt.NDArray:
    """Deformation correction for volume-deform data

    The contour in RT-DC measurements is computed on a
    pixelated grid. Due to sampling problems, the measured
    deformation is overestimated and must be corrected.

    The correction is derived in scripts/pixelation_correction.py.

    Parameters
    ----------
    volume
        The "volume" feature (rotation of raw contour) [µm³]
    px_um
        The detector pixel size in µm.

    Returns
    -------
    deform_delta
        Error of the deformation of the event(s) that must be
        subtracted from `deform`.
        deform_corr = deform -  deform_delta
    """
    pxscalev = (.34 / px_um)**3
    offs = 0.0013
    exp1 = 0.0172 * np.exp(-volume * pxscalev / 40)
    exp2 = 0.0070 * np.exp(-volume * pxscalev / 450)
    exp3 = 0.0032 * np.exp(-volume * pxscalev / 6040)
    delta = offs + exp1 + exp2 + exp3
    return delta


def get_pixelation_delta_pair(
        feat1: str,
        feat2: str,
        data1: float | npt.NDArray,
        data2: float | npt.NDArray,
        px_um: float = 0.34) -> (tuple[float, float] |
                                 tuple[npt.NDArray, npt.NDArray]):
    """Convenience function that returns pixelation correction pair"""
    # determine feature that defines abscissa
    feat_absc = feat1 if feat1 in ["area_um", "volume"] else feat2
    data_absc = data1 if feat_absc == feat1 else data2
    # first compute all the possible pixelation offsets
    delt1 = get_pixelation_delta(
        feat_corr=feat1,
        feat_absc=feat_absc,
        data_absc=data_absc,
        px_um=px_um)
    delt2 = get_pixelation_delta(
        feat_corr=feat2,
        feat_absc=feat_absc,
        data_absc=data_absc,
        px_um=px_um)
    return delt1, delt2


def get_pixelation_delta(feat_corr: str,
                         feat_absc: str,
                         data_absc: float | npt.NDArray,
                         px_um: float = 0.34) -> float | npt.NDArray:
    """Convenience function for obtaining pixelation correction

    Parameters
    ----------
    feat_corr
        Feature for which to compute the pixelation correction
        (e.g. "deform")
    feat_absc
        Feature with which to compute the correction (e.g. "area_um");
    data_absc
        Corresponding data for `feat_absc`
    px_um: float
        Detector pixel size [µm]
    """
    if feat_corr == "deform" and feat_absc == "area_um":
        delt = corr_deform_with_area_um(data_absc, px_um=px_um)
    elif feat_corr == "circ" and feat_absc == "area_um":
        delt = -corr_deform_with_area_um(data_absc, px_um=px_um)
    elif feat_corr == "deform" and feat_absc == "volume":
        delt = corr_deform_with_volume(data_absc, px_um=px_um)
    elif feat_corr == "circ" and feat_absc == "volume":
        delt = -corr_deform_with_volume(data_absc, px_um=px_um)
    elif feat_corr == "area_um":
        # no correction for area
        delt = np.zeros_like(data_absc, dtype=float)
    elif feat_corr == "volume":
        # no correction for volume
        delt = np.zeros_like(data_absc, dtype=float)
    elif feat_corr == feat_absc:
        raise ValueError("Input feature names are identical!")
    else:
        raise KeyError(
            "No rule for feature '{}' with abscissa ".format(feat_corr)
            + "'{}'!".format(feat_absc))
    return delt
