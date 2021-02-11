"""
A collection of utility functions used by SYD.py.
"""

from collections import deque
from typing import Tuple

import numpy as np
from astropy.io import ascii
from astropy.convolution import (Box1DKernel, Gaussian1DKernel, convolve,
                                 convolve_fft)
from scipy.ndimage import filters
from scipy.special import erf

from constants import *


##########################################################################################
#                                                                                        #
#                                   DIFFERENT MODELS                                     #
#                                                                                        #
##########################################################################################


def power_law(frequency, pars, compare=False, power=None, error=None):
    """
    Power law model.
    """

    model = np.array([pars[0]/(f**pars[1]) for f in frequency])
    if compare:
        return (power - model)/error
    else:
        return model


def lorentzian(_frequency, pars, _compare=False, _power=None, _error=None):
    """
    Lorentzian model.
    """
    model = np.array([pars])

    return model


def harvey(frequency, parameters, mode="regular", gaussian_model=False, total=False):
    """Harvey model.

    Parameters
    ----------
    frequency : np.ndarray
        frequencies
    parameters : list
        Harvey model parameters
    mode : str
        regular mode , second mode and fourth mode
    gaussian_model : bool
        flag to apply Gaussian skew to model
    total : bool
        flag that TODO: Not sure what this exactly means
    """

    if gaussian_model:
        nlaws = int((len(parameters)-6)/2.0)
    else:
        nlaws = int((len(parameters)-1)/2.0)
    model = np.zeros_like(frequency)

    # Sums over all power laws
    if mode == "regular":
        for i in range(nlaws):
            model += parameters[i * 2]/(1.0 + (parameters[(i * 2) + 1] * frequency)**2.0 + (parameters[(i * 2) + 1] * frequency)**4.0)
    # Sum only over second power law
    elif mode == "second":
        for i in range(nlaws):
            model += parameters[i * 2]/(1.0 + (parameters[(i * 2) + 1] * frequency)**2.0)
    # Sum only over fourth power law
    elif mode == "fourth":
        for i in range(nlaws):
            model += parameters[i * 2]/(1.0 + (parameters[(i * 2) + 1] * frequency)**4.0)
    else:
        print("Wrong mode input for the harvey model function.")

    if gaussian_model:
        model += gaussian_skew(frequency, parameters[2*nlaws + 1:])
    if total:
        model += parameters[2*nlaws]
    return model


def generate_model(frequency, pars, pars_errs, nyquist):
    """
    Generates model.
    """
    
    ps = np.zeros_like(frequency)

    for i, f in enumerate(frequency):

        r = (np.sin((np.pi*f)/(2.0*nyquist))/((np.pi*f)/(2.0*nyquist)))**2
        paras = [p+np.random.randn()*p_e for p, p_e in zip(pars, pars_errs)]
        nlaws = int((len(paras)-1.0)/2.0)
        m = 0
        for j in range(nlaws):
            m += paras[j*2]/(1.0+(paras[(j*2)+1]*f)**2.0+(paras[(j*2)+1]*f)**4.0)
        m *= r
        m += pars[-1] + np.random.random_integers(-1, 1)*(pars[-1]/2.0)**(np.random.randn()-1.)
        if m < 0.:
            m = (10**(np.random.randn()))*r
        ps[i] = m

    return list(ps)


def gaussian(frequency: np.ndarray, offset: float, amplitude: float, center: float, width: float) -> np.ndarray:
    """General Gaussian function.

    Parameters:
    frequency : np.ndarray
        power spectrum frequencies
    offset : float
        vertical offset
    amplitude : float
        amplitude
    center : float
        center of the peak or the horizontal offset
    width : float
        width of the peak

    Returns
    -------
    Gaussian of frequency
    """

    return offset + amplitude * np.exp((-(center - frequency)**2.0)/(2.0 * width**2.0))


def harvey_one(frequency: np.ndarray, a1: float, b1: float, white_noise: float) -> np.ndarray:
    """First Harvey model.
    
    Parameters
    ----------
    frequency : np.ndarray
        power spectrum frequencies
    a1 : float
        represents `4 * sigma**2 * tau`
    b1 : float
        represents `2 * pi * tau`
    white_noise : float
        white noise component
    
    Returns
    -------
    model : np.ndarray
        first Harvey model
    """

    model = np.zeros_like(frequency)

    model += a1 / (1.0 + (b1 * frequency)**2.0 + (b1 * frequency)**4.0)
    model += white_noise

    return model


def harvey_two(frequency: np.ndarray, a1: float, b1: float, a2: float, b2: float, white_noise: float) -> np.ndarray:
    """Second Harvey model.

    Parameters
    ----------
    frequency : np.ndarray
        power spectrum frequencies
    a1 : float
        represents `4 * sigma1**2 * tau1`
    b1 : float
        represents `2 * pi * tau1`
    a2 : float
        represents `4 * sigma2**2 * tau2`
    b2 : float
        represents `2 * pi * tau2`
    white_noise : float
        white noise component

    Returns
    -------
    model : np.ndarray
        second Harvey model
    """

    model = np.zeros_like(frequency)

    model += a1 / (1.0 + (b1 * frequency)**2.0 + (b1 * frequency)**4.0)
    model += a2 / (1.0 + (b2 * frequency)**2.0 + (b2 * frequency)**4.0)
    model += white_noise

    return model


def harvey_three(frequency: np.ndarray, a1: float, b1: float, a2: float, b2: float, a3: float, b3: float, white_noise: float) -> np.ndarray:
    """Third Harvey model.

    Parameters
    ----------
    frequency : np.ndarray
        power spectrum frequencies
    a1 : float
        represents `4 * sigma1**2 * tau1`
    b1 : float
        represents `2 * pi * tau1`
    a2 : float
        represents `4 * sigma2**2 * tau2`
    b2 : float
        represents `2 * pi * tau2`
    a3 : float
        represents `4 * sigma3**2 * tau3`
    b3 : float
        represents `2 * pi * tau3`
    white_noise : float
        white noise component

    Returns
    -------
    model : np.ndarray
        third Harvey model
    """

    model = np.zeros_like(frequency)

    model += a1 / (1.0 + (b1 * frequency)**2.0 + (b1 * frequency)**4.0)
    model += a2 / (1.0 + (b2 * frequency)**2.0 + (b2 * frequency)**4.0)
    model += a3 / (1.0 + (b3 * frequency)**2.0 + (b3 * frequency)**4.0)
    model += white_noise

    return model


def gaussian_skew(frequency, parameters):
    """Gaussian skew model.
    
    Parameters
    ----------
    frequency : np.ndarray
        frequencies
    parameters : list
        function parameters
    """
    
    # TODO: Delete comment
    # Don't understand how this even works. Gaussian isn't supplied enough arguments and I'm not sure where x gets defined.
    model = np.array(
        [2.0 * gaussian(f, parameters[0:4]) * 0.5*(1.0 + erf(parameters[4] * ((frequency - parameters[1])/parameters[2])/np.sqrt(2.0))) for f in frequency]
    )

    return model


##########################################################################################
#                                                                                        #
#                                DATA MANIPULATION ROUTINES                              #
#                                                                                        #
##########################################################################################


def mean_smooth_ind(x: np.ndarray, y: np.ndarray, width: float):
    """Smooths the data using independent mean smoothing and binning.

    Parameters
    ----------
    x : np.ndarray
        x data
    y : np.ndarray
        y data
    width : float
        independent average smoothing width

    Returns
    -------
    smooth_x : np.ndarray
        binned mean smoothed x data
    smooth_y : np.ndarray
        binned mean smoothed y data
    standard_error : np.ndarray
        standard error
    """

    step = width - 1
    smooth_x = np.zeros_like(x)
    smooth_y = np.zeros_like(x)
    standard_error = np.zeros_like(x)

    j = 0

    while (j + step) < (len(x) - 1):
        # Replace with mean
        smooth_x[j] = np.mean(x[j:j+step])
        smooth_y[j] = np.mean(y[j:j+step])
        # Compute standard error
        standard_error[j] = np.std(y[j:j+step])/np.sqrt(width)
        # Jump to next index by width-1
        j += step

    smooth_x = smooth_x[(smooth_x != 0.0)]
    standard_error = standard_error[(smooth_y != 0.0)]
    smooth_y = smooth_y[(smooth_y != 0.0)]
    standard_error[(standard_error == 0.0)] = np.median(standard_error)

    return smooth_x, smooth_y, standard_error


def bin_data(frequency: np.ndarray, power: np.ndarray, binning: float) -> Tuple[np.ndarray, np.ndarray]:
    """Bins data logarithmically.

    Parameters
    ----------
    frequency : np.ndarray
        power spectrum frequency
    power : np.ndarray
        power spectrum power
    binning : float
        logarithmic binning width

    Returns
    -------
    bin_frequency : np.ndarray
        binned frequencies
    bin_power : np.ndarray
        binned power
    """

    min_log = np.log10(min(frequency))
    max_log = np.log10(max(frequency))
    bin_number = int(np.ceil((max_log - min_log)/binning))
    bins = np.logspace(min_log, min_log + (bin_number + 1)*binning, bin_number)

    # Bin indices
    digitized = np.digitize(frequency, bins)
    # Bin frequency and power
    bin_frequency = np.array(
        [frequency[digitized == i].mean() for i in range(1, len(bins)) if len(frequency[digitized == i]) > 0]
    )
    bin_power = np.array(
        [power[digitized == i].mean() for i in range(1, len(bins)) if len(power[digitized == i]) > 0]
    )

    return bin_frequency, bin_power


def smooth(array, width, params, method="box", mode=None, fft=False, silent=False):
    """
    Smooths using a variety of methods.
    """

    if method == "box":

        if isinstance(width, int):
            kernel = Box1DKernel(width)
        else:
            width = int(np.ceil(width/params["resolution"]))
            kernel = Box1DKernel(width)

        if fft:
            smoothed_array = convolve_fft(array, kernel)
        else:
            smoothed_array = convolve(array, kernel)

        if not silent:
            print(f"{method} kernel: kernel size = {width*params['resolution']:.2f} muHz")

    elif method == "gaussian":

        n = 2*len(array)
        forward = array[:].tolist()
        reverse = array[::-1].tolist()

        if n%4 != 0:
            start = int(np.ceil(n/4))
        else:
            start = int(n/4)
        end = len(array)

        final = np.array(reverse[start:end]+forward[:]+reverse[:start])

        if isinstance(width, int):
            kernel = Gaussian1DKernel(width)
        else:
            width = int(np.ceil(width/params["resolution"]))
            kernel = Gaussian1DKernel(width, mode = mode)

        if fft:
            smoothed = convolve_fft(final, kernel)
        else:
            smoothed = convolve(final, kernel)

        smoothed_array = smoothed[int(n/4):int(3*n/4)]

        if not silent:
            print("%s kernel: sigma = %.2f muHz"%(method, width*params["resolution"]))
            print(f"{method} kernel: sigma = {width*params['resolution']:.2f} muHz")
    else:
        print("Do not understand the smoothing method chosen.")
        smoothed_array = None

    return smoothed_array


def max_elements(array, N, resolution, limit=[False, None]):
    """
    Returns the indices of the maximum elements.
    """

    indices = []

    while len(indices) < N:

        new_max = max(array)
        idx = array.index(new_max)
        add = True
        if indices != [] and limit[0]:
            for index in indices:
                if np.absolute((index - idx)*resolution) < limit[1]:
                    add = False
                    break
        if add:
            indices.append(idx)
        array[idx] = 0.

    return np.array(indices)


def smooth_gauss(array, fwhm, params, silent = False):
    """
    Smooths using Gaussian convolution.
    """

    sigma = fwhm/np.sqrt(8.*np.log(2.))

    n = 2*len(array)
    N = np.arange(1,n+1,1)
    mu = len(array)
    total = np.sum((1./(sigma*np.sqrt(2.*np.pi)))*np.exp(-0.5*(((N-mu)/sigma)**2.)))
    weights = ((1./(sigma*np.sqrt(2.*np.pi)))*np.exp(-0.5*(((N-mu)/sigma)**2.)))/total

    forward = array[:]
    reverse = array[::-1]

    if n % 4 != 0:
        start = int(np.ceil(n/4))
    else:
        start = int(n/4)
    end = int(n/2)

    final = np.array(reverse[start:end]+forward[:]+reverse[:start])
    fft = np.fft.irfft(np.fft.rfft(final)*np.fft.rfft(weights))
    dq = deque(fft)
    dq.rotate(int(n/2))
    smoothed = np.array(dq)
    smoothed_array = smoothed[int(n/4):int(3*n/4)]
    if not silent:
        print("gaussian kernel using ffts: sigma = %.2f muHz"%(sigma*params["resolution"]))
    if params["edge"][0]:
        smoothed_array = smoothed_array[:-params["edge"][1]]

    return np.array(smoothed_array)


def corr(frequency, power, params):
    """
    Calculates the correlation.
    """

    _f = frequency[:]
    p = power[:]
        
    n = len(p)
    mean = np.mean(p)
    _var = np.var(p)   
    N = np.arange(n)
    
    lag = N*params["resolution"]
    
    auto = np.correlate(p - mean, p - mean, "full")    
    auto = auto[int(auto.size/2):]

    mask = np.ma.getmask(np.ma.masked_inside(lag, params["fitbg"]["lower_lag"], params["fitbg"]["upper_lag"]))
    
    lag = lag[mask]
    auto = auto[mask]
    
    return lag, auto


def estimate_mass(radius: float, logg: float) -> float:
    """Estimates stellar mass using stellar radius and surface gravity.

    Parameters
    ----------
    radius : float
        the radius of the star in solRad
    logg : float
        the log of surface gravity

    Returns
    -------
    mass : float
        the estimated mass of the star in solMass
    """

    return ((radius * RADIUS_SUN)**2.0 * 10**(logg) / G) / MASS_SUN


def estimate_numax(radius: float, mass: float, teff: float) -> float:
    """Estimates numax using stellar radius, stellar mass and effective temperature.

    Parameters
    ----------
    radius : float
        the radius of the star in solRad
    mass : float
        the mass of the star in solMass
    teff : float
        the effective temperature of the star in Kelvin

    Returns
    -------
    numax : float
        the estimated numax of the star in muHz
    """

    return NUMAX_SUN * (radius**(-2.0)) * mass * ((teff/TEFF_SUN)**(-0.5))


def estimate_deltanu(radius: float, mass: float) -> float:
    """Estimates delta nu using stellar radius and stellar mass.

    Parameters
    ----------
    radius : float
        the radius of the star in solRad
    mass : float
        the mass of the star in solMass

    Returns
    -------
    deltanu : float
        the estimated deltanu of the star in muHz
    """

    return DNU_SUN * (radius**(-1.5)) * (mass**0.5)


def estimate_delta_nu_from_numax(numax: float) -> float:
    """Estimates delta nu using a numax scaling relation.

    Parameters
    ----------
    numax : float
        numax frequency

    Returns
    -------
    delta_nu : float
        delta_nu frequency separation
    """

    return 0.22 * (numax**0.797)
