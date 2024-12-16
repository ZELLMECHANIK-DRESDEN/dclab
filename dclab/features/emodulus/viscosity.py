"""Viscosity computation for various media"""
from __future__ import annotations

from typing import Literal
import warnings

import numpy as np
import numpy.typing as npt

from ...warn import PipelineWarning


#: Dictionary with different names for one medium
SAME_MEDIA = {
    "0.49% MC-PBS": ["0.49% MC-PBS",
                     "0.5% MC-PBS",
                     "0.50% MC-PBS",
                     "CellCarrier",
                     ],
    # 6g MC in 1010g PBS [!sic]
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
                      temperature: float | npt.NDArray,
                      tmin: float,
                      tmax: float) -> None:
    """Raise a TemperatureOutOfRangeWarning if applicable

    Raises
    ------
    TemperatureOutOfRangeWarning
        If the given temperature is out of the given range.

    """
    if np.min(temperature) < tmin or np.max(temperature) > tmax:
        warnings.warn(
            f"For the {model} model, the temperature should be "
            f"in [{tmin}, {tmax}] degC! Got min/max of "
            f"[{np.min(temperature):.1f}, {np.max(temperature):.1f}] degC.",
            TemperatureOutOfRangeWarning)


def get_viscosity(medium: str = "0.49% MC-PBS",
                  channel_width: float = 20.0,
                  flow_rate: float = 0.16,
                  temperature: float | npt.NDArray = 23.0,
                  model: Literal['herold-2017',
                                 'herold-2017-fallback',
                                 'buyukurganci-2022',
                                 'kestin-1978'] = 'herold-2017-fallback'
                  ) -> float | npt.NDArray:
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
    medium
        The medium to compute the viscosity for; Valid values
        are defined in :const:`KNOWN_MEDIA`.
    channel_width
        The channel width in µm
    flow_rate
        Flow rate in µL/s
    temperature
        Temperature in °C
    model
        The model name to use for computing the medium viscosity.
        For water, this value is ignored, as there is only the
        'kestin-1978' model :cite:`Kestin_1978`. For MC-PBS media,
        there are the 'herold-2017' model :cite:`Herold2017` and the
        'buyukurganci-2022' model :cite:`Buyukurganci2022`.

    Returns
    -------
    viscosity
        Viscosity in mPa*s

    Notes
    -----
    - CellCarrier (0.49% MC-PBS) and CellCarrier B (0.59% MC-PBS) are
      media designed for RT-DC experiments.
    - A :class:`TemperatureOutOfRangeWarning` is issued if the
      input temperature range exceeds the temperature ranges of
      the models.
    """
    # also support lower-case media and a space before the "B"
    if medium not in KNOWN_MEDIA:
        raise ValueError(f"Invalid medium: {medium}")
    medium = ALIAS_MEDIA[medium]

    if medium == "water":
        # We ignore the `model`, because it's user convenient.
        eta = get_viscosity_water_kestin_1978(temperature=temperature)
    elif medium in ["0.49% MC-PBS", "0.59% MC-PBS", "0.83% MC-PBS"]:
        kwargs = {"medium": medium,
                  "temperature": temperature,
                  "flow_rate": flow_rate,
                  "channel_width": channel_width}

        # Let the user know that we have a new model in town.
        if model == "herold-2017-fallback":
            warnings.warn(
                "dclab 0.48.0 introduced a more accurate model for computing "
                "the MC-PBS viscosity. You are now using the old model "
                "'herold-2017'. Unless you are reproducing an old analysis "
                "pipeline, you should consider passing 'buyukurganci-2022' "
                "as a viscosity model!",
                DeprecationWarning)
            model = "herold-2017"

        if model == "herold-2017":
            eta = get_viscosity_mc_pbs_herold_2017(**kwargs)
        elif model == "buyukurganci-2022":
            eta = get_viscosity_mc_pbs_buyukurganci_2022(**kwargs)
        else:
            raise NotImplementedError(f"Unknown model '{model}' for MC-PBS!")
    else:
        raise NotImplementedError(f"Unknown medium '{medium}'!")
    return eta


def shear_rate_square_channel(flow_rate: float,
                              channel_width: float,
                              flow_index: float) -> float:
    """Returns The wall shear rate of a power law liquid in a squared channel.

    Parameters
    ----------
    flow_rate
        Flow rate in µL/s
    channel_width
        The channel width in µm
    flow_index
        Flow behavior index aka the power law exponent of the shear thinning

    Returns
    -------
    shear_rate
        Shear rate in 1/s.
    """
    # convert channel width to mm
    channel_width = channel_width * 1e-3
    return 8*flow_rate/(channel_width**3) * (0.6671 + 0.2121/flow_index)


def get_viscosity_mc_pbs_buyukurganci_2022(
        medium: Literal["0.49% MC-PBS",
                        "0.59% MC-PBS",
                        "0.83% MC-PBS"] = "0.49% MC-PBS",
        channel_width: float = 20.0,
        flow_rate: float = 0.16,
        temperature: float = 23.0) -> float | npt.NDArray:
    """Compute viscosity of MC-PBS according to :cite:`Buyukurganci2022`

    This viscosity model was derived in :cite:`Buyukurganci2022`
    and adapted for RT-DC in :cite:`Reichel2023`.
    """
    check_temperature("'buyukurganci-2022' MC-PBS", temperature, 22, 37)
    # material constants for temperature behavior of MC dissolved in PBS:
    alpha = 0.00223
    lambd = 3379.7
    kelvin = temperature + 273.15

    if medium == "0.49% MC-PBS":
        a = 2.30e-6  # previously 2.23e-6, changed in Reichel2023 rev 2
        beta = -0.0056
    elif medium == "0.59% MC-PBS":
        a = 5.70e-6
        beta = -0.0744
    elif medium == "0.83% MC-PBS":
        a = 16.52e-6
        beta = -0.1455
    else:
        raise NotImplementedError(
            f"Medium {medium} not supported for model `buyukurganci-2022`!")

    k = a * np.exp(lambd / kelvin)
    n = alpha * kelvin + beta
    shear_rate = shear_rate_square_channel(flow_rate, channel_width, n)
    return k * shear_rate**(n - 1) * 1e3


def get_viscosity_mc_pbs_herold_2017(
        medium: Literal["0.49% MC-PBS", "0.59% MC-PBS"] = "0.49% MC-PBS",
        channel_width: float = 20.0,
        flow_rate: float = 0.16,
        temperature: float = 23.0) -> float | npt.NDArray:
    r"""Compute viscosity of MC-PBS according to :cite:`Herold2017`

    Note that all the factors in equation 5.2 in :cite:`Herold2017`
    compute to 8, which is essentially what is implemented in
    :func:`shear_rate_square_channel`:

    .. math::

        1.1856 \cdot 6 \cdot \frac{2}{3} \cdot \frac{1}{0.5928} = 8
    """
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
            f"Medium {medium} not supported for model `herold-2017`!")
    return eta


def get_viscosity_water_kestin_1978(
        temperature: float = 23.0) -> float | npt.NDArray:
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
