"""Viscosity computation for various media"""
from __future__ import annotations

from typing import Literal
import warnings

import numpy as np

from ...warn import PipelineWarning


#: Dictionary with different names for one medium
SAME_MEDIA = {
    "0.49% MC-PBS": ["0.49% MC-PBS",
                     "0.5% MC-PBS",
                     "0.50% MC-PBS",
                     "CellCarrier",
                     ],
    "0.59% MC-PBS": ["0.59% MC-PBS",
                     "0.6% MC-PBS",
                     "0.60% MC-PBS",
                     "CellCarrier B",
                     "CellCarrierB",
                     ],
    "0.83% MC-PBS": ["0.83% MC-PBS",
                     "0.8% MC-PBS",
                     "0.80% MC-PBS"],
    "water": ["water"],
}

#: Many media names are actually shorthand for one medium
ALIAS_MEDIA = {}
for key in SAME_MEDIA:
    for item in SAME_MEDIA[key]:
        ALIAS_MEDIA[item] = key
        ALIAS_MEDIA[item.lower()] = key  # also support all-lower case

#: Media for which computation of viscosity is defined (has duplicate entries)
KNOWN_MEDIA = sorted(ALIAS_MEDIA.keys())


class TemperatureOutOfRangeWarning(PipelineWarning):
    pass


def check_temperature(model: str,
                      temperature: float | np.array,
                      tmin: float,
                      tmax: float):
    """Raise a TemperatureOutOfRangeWarning if applicable"""
    if np.min(temperature) < tmin or np.max(temperature) > tmax:
        warnings.warn(
            f"For the {model} model, the temperature should be "
            f"in [{tmin}, {tmax}] degC! Got min/max of "
            f"[{np.min(temperature):.1f}, {np.max(temperature):.1f}] degC.",
            TemperatureOutOfRangeWarning)


def get_viscosity(medium: str = "0.49% MC-PBS",
                  channel_width: float = 20.0,
                  flow_rate: float = 0.16,
                  temperature: float = 23.0,
                  model: Literal['herold-2017',
                                 'buyukurganci-2022',
                                 'kestin-1978'] = 'herold-2017'):
    """Returns the viscosity for RT-DC-specific media

    Media that are not pure (e.g. ketchup or polymer solutions)
    often exhibit a non-linear relationship between shear rate
    (determined by the velocity profile) and shear stress
    (determined by pressure differences). If the shear stress
    grows non-linearly with the shear rate resulting in a slope
    in log-log space that is less than one, then we are talking about
    shear thinning. The viscosity is not a constant anymore (as it
    is e.g. for water). At higher flow rates, the viscosity becomes
    smaller, following a power law. Christoph Herold characterized
    shear thinning for the CellCarrier media :cite:`Herold2017`.
    The resulting formulae for computing the viscosities of these
    media at different channel widths, flow rates, and temperatures,
    are implemented here.

    Parameters
    ----------
    medium: str
        The medium to compute the viscosity for; Valid values
        are defined in :const:`KNOWN_MEDIA`.
    channel_width: float
        The channel width in µm
    flow_rate: float
        Flow rate in µL/s
    temperature: float or ndarray
        Temperature in °C
    model: str
        The model name to use for computing the medium viscosity.
        For water, this value is ignored, as there is only the
        'kestin-1978' model :cite:`Kestin_1978`. For MC-PBS media,
        there are the 'herold-2017' model :cite:`Herold2017` and the
        'buyukurganci-2022' model :cite:`Buyukurganci2022`.

    Returns
    -------
    viscosity: float or ndarray
        Viscosity in mPa*s

    Notes
    -----
    - CellCarrier and CellCarrier B media are optimized for
      RT-DC measurements.
    - A :class:`TemperatureOutOfRangeWarning` is issued if the
      input temperature range exceeds the temperature ranges of
      the models.
    """
    # also support lower-case media and a space before the "B"
    if medium not in KNOWN_MEDIA:
        raise ValueError(f"Invalid medium: {medium}")
    medium = ALIAS_MEDIA[medium]

    if medium == "water":
        eta = get_viscosity_water_kestin_1978(temperature=temperature)
    elif medium in ["0.49% MC-PBS", "0.59% MC-PBS", "0.83% MC-PBS"]:
        kwargs = {"medium": medium,
                  "temperature": temperature,
                  "flow_rate": flow_rate,
                  "channel_width": channel_width}
        if model == "herold-2017":
            eta = get_viscosity_mc_pbs_herold_2017(**kwargs)
        elif model == "buyukurganci-2022":
            eta = get_viscosity_mc_pbs_buyukurganci_2022(**kwargs)
        else:
            raise NotImplementedError(f"Unknown model '{model}' for MC-PBS!")
    else:
        raise NotImplementedError(f"Unknown medium '{medium}'!")
    return eta


def get_viscosity_mc_pbs_buyukurganci_2022(
        medium: Literal["0.49% MC-PBS",
                        "0.59% MC-PBS",
                        "0.83% MC-PBS"] = "0.49% MC-PBS",
        channel_width: float = 20.0,
        flow_rate: float = 0.16,
        temperature: float = 23.0):
    """Compute viscosity of MC-PBS according to :cite:`Buyukurganci2022`"""
    check_temperature("'buyukurganci-2022' MC-PBS", temperature, 22, 37)
    raise NotImplementedError("Model buyukurganci-2022 not implemented yet!")


def get_viscosity_mc_pbs_herold_2017(
        medium: Literal["0.49% MC-PBS", "0.59% MC-PBS"] = "0.49% MC-PBS",
        channel_width: float = 20.0,
        flow_rate: float = 0.16,
        temperature: float = 23.0):
    """Compute viscosity of MC-PBS according to :cite:`Herold2017`"""
    # see figure (9) in Herold arXiv:1704.00572 (2017)
    check_temperature("'herold-2017' MC-PBS", temperature, 18, 26)
    # convert flow_rate from µL/s to m³/s
    # convert channel_width from µm to m
    term1 = 1.1856 * 6 * flow_rate * 1e-9 / (channel_width * 1e-6)**3 * 2 / 3

    if medium == "0.49% MC-PBS":
        temp_corr = (temperature / 23.2)**-0.866
        term2 = 0.6771 / 0.5928 + 0.2121 / (0.5928 * 0.677)
        eta = 0.179 * (term1 * term2)**(0.677 - 1) * temp_corr * 1e3
    elif medium == "0.59% MC-PBS":
        temp_corr = (temperature / 23.6)**-0.866
        term2 = 0.6771 / 0.5928 + 0.2121 / (0.5928 * 0.634)
        eta = 0.360 * (term1 * term2)**(0.634 - 1) * temp_corr * 1e3
    else:
        raise NotImplementedError(
            f"Medium {medium} not supported for model `herold2017`!")
    return eta


def get_viscosity_water_kestin_1978(temperature: float = 23.0):
    """Compute the viscosity of water according to :cite:`Kestin_1978`"""
    # see equation (15) in Kestin et al, J. Phys. Chem. 7(3) 1978
    check_temperature("'kestin-1978' water", temperature, 0, 40)
    eta0 = 1.002  # [mPa s]
    right = (20-temperature) / (temperature + 96) \
        * (+ 1.2364
           - 1.37e-3 * (20 - temperature)
           + 5.7e-6 * (20 - temperature)**2
           )
    eta = eta0 * 10**right
    return eta
