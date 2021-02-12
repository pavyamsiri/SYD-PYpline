"""
Perform a fit to the granulation background and measures the frequency of maximum power (nu_max),
the large frequency separation (Delta_nu) and oscillation amplitude.
"""

import os
from typing import Dict, Iterable, Tuple

import numpy as np
from astropy.convolution import (Box1DKernel, Gaussian1DKernel, convolve,
                                 convolve_fft)
from astropy.stats import mad_std
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm
from scipy.optimize import curve_fit
from scipy.stats import chisquare

from constants import *
from functions import *

##########################################################################################
#                                                                                        #
#                                  FIT BACKGROUND ROUTINE                                #
#                                                                                        #
##########################################################################################
# TODOS
# 1) Change the way the n_laws is modified within the code (i.e. drop Faculae term, long period trends)
# 2) Making sure the correct number of harvey components make it to the final fit
# ADDED
# 1) Ability to change number of points used to calculate RMS of harvey component (default = 10)


class FitBackground:
    """Main class that handles the execution of the fit background routine.
    
    TODO: Fill out attributes
    """
    
    def __init__(
            self,
            save: bool = True,
            verbose: bool = False,
            show_plots: bool = False,
            num_mc_iter: int = 200,
            lower: float = 10.0,
            upper: float = None,
            box_filter: float = 1.0,
            smoothing_frequency: float = 1.0,
            ind_width: int = 50,
            n_rms: int = 20,
            n_peaks: int = 5,
            force: bool = False,
            guess: float = 140.24,
            clip: bool = True,
            clip_value: float = 0.0,
            lower_numax: float = None,
            upper_numax: float = None,
            echelle_smooth: bool = False,
            echelle_filter: float = 1.0,
    ) -> None:
        """Initialises parameters needed for the fit background routine.
        
        Parameters
        ----------
        save : bool
            flag to save results
        verbose : bool
            flag to increase verbosity
        show_plots : bool
            flag to show plots
        num_mc_iter : int
            number of Markov-Chain iterations TODO: Markov-Chain or Monte-Carlo?
        lower: float
            lower frequency bound
        upper: float
            upper frequency bound
        box_filter : float
            size of the box filter
        smoothing_frequency : float
            frequency for smoothing power spectrum in muHz
        ind_width : int
            independent average smoothing width
        n_rms : int
            rms intensity TODO: Not sure if this is correct
        n_peaks : int
            is it to do with the number of highlighted peaks in the power spectrum TODO: Don't know
        force : bool
            TODO: Don't know
        guess : float
            TODO: Don't know
        clip : bool
            TODO: Don't know
        clip_value : float
            TODO: Don't know
        lower_numax : float
            lower bound for numax
        upper_numax : float
            upper bound for numax
        echelle_smooth : bool
            TODO: Don't know
        echelle_filter : float
            TODO: Don't know
        """

        self.flags: Dict[str, bool] = {
            "save": save,
            "verbose": verbose,
            "show_plots": show_plots,
            "force": force,
            "clip": clip,
            "echelle_smooth": echelle_smooth
        }
        self.num_mc_iter: int = num_mc_iter
        self.lower: float = lower
        self.upper: float = upper
        self.box_filter: float = box_filter
        self.smoothing_frequency: float = smoothing_frequency
        self.ind_width: int = ind_width
        self.n_rms: int = n_rms
        self.n_peaks: int = n_peaks
        self.lower_numax: float = lower_numax
        self.upper_numax: float = upper_numax
        self.guess: float = guess
        self.clip_value: float = clip_value
        self.echelle_filter: float = echelle_filter

        # Harvey models
        self.functions = {
            1: harvey_one,
            2: harvey_two,
            3: harvey_three,
        }

        # Uninitialised target data
        self.target: str = None
        self.path: str = None
        # Time series
        self.time: np.ndarray = None
        self.flux: np.ndarray = None
        self.resolution: float = None
        self.nyquist: float = None
        # Power spectrum
        self.frequency: np.ndarray = None
        self.power: np.ndarray = None
        self.envelope_edges: Tuple[float, float] = None
        self.smooth_power: np.ndarray = None
        self.mask: np.ndarray = None
        
        # Harvey models
        self.fitted_harvey_parameters: list = None
        self.final_parameters: list = None
        self.num_laws: int = None
        self.num_laws_original: int = None
        self.noise: float = None
        # Binned power spectrum
        self.binned_frequency: np.ndarray = None
        self.binned_power: np.ndarray = None
        self.binned_error: np.ndarray = None
        # TODO: rename
        self.mnu_original: np.ndarray = None
        self.a_original: np.ndarray = None
        # Background fit
        self.pssm: np.ndarray = None
        # Smoothed power excess with Gaussian
        self.region_frequency: np.ndarray = None
        self.region_power: np.ndarray = None
        self.ps_gaussian_parameters = None
        # Background corrected power spectrum
        self.envelope_frequency: np.ndarray = None
        self.psd: np.ndarray = None

        # dnu ACFs
        self.lag: np.ndarray = None
        self.auto: np.ndarray = None
        self.lag_peaks: np.ndarray = None
        self.auto_peaks: np.ndarray = None
        self.best_lag: float = None
        self.best_auto: float = None
        self.zoom_lag: np.ndarray = None
        self.zoom_auto: np.ndarray = None
        self.dnu_gaussian_parameters: list = None
        
        # echelle TODO: finish type hints
        self.echelle = None
        self.echelle_copy = None
        self.extent = None
        self.mod_frequency = None
        self.collapsed_power = None

        # numax and dnu
        self.numax: float = None
        self.dnu: float = None
        self.snr: float = 0
        self.width: float = 0
        self.num_dnus: float = 0

    def update_target(
            self,
            target: str,
            path: str,
            time: np.ndarray,
            flux: np.ndarray,
            frequency: np.ndarray,
            power: np.ndarray,
            resolution: float,
            nyquist: float,
            # numax: float,
            # dnu: float,
            # snr: float,
            # width: float,
            # num_dnus: float,
    ) -> None:
        """Updates current target.

        Parameters
        ----------
        target : str
            target to be processed
        path : str
            target path
        time : np.ndarray
            light curve times
        flux : np.ndarray
            light curve fluxes
        frequency : np.ndarray
            power spectrum frequencies
        power : np.ndarray
            power spectrum power
        resolution : float
            frequency resolution of power spectrum
        nyquist : float
            Nyquist frequency of the time series
        numax : float
            prior numax frequency
        dnu : float
            prior dnu frequency spacing
        snr : float
            snr of the prior numax estimate
        width : float
            width of the power envelope
        num_dnus : float
            number of large frequency spacings in the power envelope
        """

        # Uninitialised target data
        self.target = target
        self.path = path
        # Time series
        self.time = time
        self.flux = flux
        self.resolution = resolution
        self.nyquist = nyquist
        # Power spectrum
        self.frequency = frequency
        self.power = power
        self.envelope_edges = None
        self.smooth_power = None
        self.mask = None

        # Harvey models
        self.fitted_harvey_parameters = None
        self.final_parameters = None
        self.num_laws = None
        self.num_laws_original = None
        self.noise = None
        # Binned power spectrum
        self.binned_frequency = None
        self.binned_power = None
        self.binned_error = None
        # TODO: rename
        self.mnu_original = None
        self.a_original = None
        # Background fit
        self.pssm = None
        # Smoothed power excess with Gaussian
        self.region_frequency = None
        self.region_power = None
        self.ps_gaussian_parameters = None
        # Background corrected power spectrum
        self.envelope_frequency = None
        self.psd = None

        # dnu ACFs
        self.lag = None
        self.auto = None
        self.lag_peaks = None
        self.auto_peaks = None
        self.best_lag: float = None
        self.best_auto: float = None
        self.zoom_lag = None
        self.zoom_auto = None
        self.dnu_gaussian_parameters: list = None
        
        # echelle TODO: finish type hints
        self.echelle = None
        self.echelle_copy = None
        self.extent = None
        self.mod_frequency = None
        self.collapsed_power = None

        # # numax and dnu
        # self.numax = numax
        # self.dnu = dnu
        # self.snr = snr
        # self.width = width
        # self.num_dnus = num_dnus

        # Make save folder
        if self.flags["save"]:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)

    def fit_background(self) -> None:
        """
        Perform a fit to the granulation background and measures the frequency of
        maximum power (numax), the large frequency separation (deltanu) and oscillation amplitude.
        """

        # Check if a prior numax exists
        if not self.check():
            return

        results = []
        # Arbitrary snr cut for leaving region out of background fit, ***statistically validate later?
        # Numax lower bound is given
        if self.lower_numax is not None and self.upper_numax is not None:
            self.envelope_edges = [self.lower_numax, self.upper_numax]
        else:
            # Low snr ratio TODO: Doesn't this require findex to be run first? snr is not part of star info
            if self.snr < 2.0:
                self.envelope_edges = [
                    self.numax - self.width/2.0,
                    self.numax + self.width/2.0
                    ]
            # High snr ratio
            else:
                self.envelope_edges = [
                    self.numax - self.num_dnus * self.dnu,
                    self.numax + self.num_dnus * self.dnu
                    ]

        # Record original PS information for plotting
        random_power = np.copy(self.power)
        bin_frequency, bin_power, bin_error = mean_smooth_ind(
            self.frequency,
            random_power,
            self.ind_width
        )

        if self.flags["verbose"]:
            print("-------------------------------------------------")
            print(f"binned to {len(bin_frequency)} data points")
        
        # Use scaling relation from Sun to get starting points
        scale = NUMAX_SUN/self.numax
        # Granulation time scales
        taus = scale * np.array(TAU_SUN)
        # Representing the term `2 * pi * tau`
        b = 2.0 * np.pi * taus * 1e-6
        mnu = (1.0/taus) * 1e5
        taus = taus[mnu >= min(self.frequency)]
        b = b[mnu >= min(self.frequency)]
        self.mnu = mnu[mnu >= min(self.frequency)]
        self.mnu_original = np.copy(self.mnu)
        # Number of power laws
        self.num_laws = len(self.mnu)
        self.num_laws_original = self.num_laws
        
        # Mask covers region of power excess
        self.mask = (self.frequency >= self.envelope_edges[0])
        self.mask &= (self.frequency <= self.envelope_edges[1])
        
        mc_iteration = 0
        # Sampling process
        while mc_iteration < self.num_mc_iter:
            print(f"{self.target}: MC iteration = {mc_iteration}")
            # Second iteration or later
            if mc_iteration > 0:
                # Randomize power spectrum to get uncertainty on measured values
                random_power = (np.random.chisquare(2, len(self.frequency)) * self.power)/2.0
                bin_frequency, bin_power, bin_error = mean_smooth_ind(
                    self.frequency,
                    random_power,
                    self.ind_width
                )

                if self.num_laws != self.num_laws_original:
                    self.mnu = self.mnu_original[:self.num_laws]
                    b = b[:self.num_laws]

            # Estimate white noise level
            self.noise = self.get_white_noise(random_power)

            # Exclude region with power excess and smooth to estimate red/white noise components
            boxkernel = Box1DKernel(int(np.ceil(self.box_filter / self.resolution)))
            outer_frequency = self.frequency[~self.mask]
            outer_power = random_power[~self.mask]
            # Smooth outer regions using convolution
            convolved_smooth_power = convolve(outer_power, boxkernel)

            # Used in curve fitting the Harvey models
            harvey_parameters = np.zeros((2*self.num_laws + 1))
            harvey_parameters[2*self.num_laws] = self.noise
            # Changes with each iteration. Represents `4 * sigma**2 * tau`
            a = np.zeros_like(self.mnu)

            # Estimate amplitude for each Harvey component
            for harvey_idx, nu in enumerate(self.mnu):
                min_idx = np.argmax(self.frequency >= nu)
                if min_idx < self.n_rms:
                    a[harvey_idx] = np.mean(convolved_smooth_power[:self.n_rms])
                elif (len(convolved_smooth_power) - min_idx) < self.n_rms:
                    a[harvey_idx] = np.mean(convolved_smooth_power[-self.n_rms:])
                else:
                    a[harvey_idx] = np.mean(convolved_smooth_power[min_idx - int(self.n_rms/2) : min_idx + int(self.n_rms/2)])

            for n in range(self.num_laws):
                harvey_parameters[2*n] = a[n]
                harvey_parameters[2*n + 1] = b[n]
            
            if mc_iteration == 0:
                bin_mask = (bin_frequency > self.envelope_edges[0]) & (bin_frequency < self.envelope_edges[1])
                self.binned_frequency = bin_frequency[~bin_mask]
                self.binned_power = bin_power[~bin_mask]
                self.binned_error = bin_error[~bin_mask]
                self.a_original = np.copy(a)
                self.smooth_power = convolve(
                    self.power,
                    Box1DKernel(int(np.ceil(self.box_filter / self.resolution)))
                )

                if self.flags["verbose"]:
                    print(f"Comparing {2 * self.num_laws} different models:")

                # Get best fit model
                bounds_list = []
                for law in range(self.num_laws):
                    # 2x3, 2x5, 2x7 ...
                    bounds = np.zeros((2, 2*(law+1) + 1)).tolist()
                    # Unbounded
                    for z in range(2*(law+1)):
                        bounds[0][z] = -np.inf
                        bounds[1][z] = np.inf
                    # Last law is bounded in an interval 0.2 wide centered around the estimated noise level
                    bounds[0][-1] = harvey_parameters[-1] - 0.1
                    bounds[1][-1] = harvey_parameters[-1] + 0.1
                    bounds_list.append(tuple(bounds))

                reduced_chi2 = []
                fitted_parameters_list = []
                # Names of the Harvey models
                names = ["one", "one", "two", "two", "three", "three"]
                model_dictionary = dict(zip(np.arange(2 * self.num_laws), names[:2*self.num_laws]))

                for model_idx in range(2 * self.num_laws):
                    # First iteration model i.e. [one, __, two, __, three, __]
                    if model_idx % 2 == 0:
                        if self.flags["verbose"]:
                            print(f"{model_idx+1}: {model_dictionary[model_idx]} harvey model w/ white noise free parameter")

                        # Get the correct number of initial variables as the current Harvey model
                        delta = 2 * (self.num_laws - (model_idx//2 + 1))
                        initial_vars = list(harvey_parameters[:(-delta-1)])
                        initial_vars.append(harvey_parameters[-1])
                        # Try to fit Harvey function
                        try:
                            fitted_vars, _cv = curve_fit(
                                self.functions[model_idx//2 + 1],
                                self.binned_frequency,
                                self.binned_power,
                                p0=initial_vars,
                                sigma=self.binned_error
                            )
                        except RuntimeError:
                            fitted_parameters_list.append([])
                            reduced_chi2.append(np.inf)
                        else:
                            fitted_parameters_list.append(fitted_vars)
                            chi, _p = chisquare(f_obs=outer_power, f_exp=harvey(outer_frequency, fitted_vars, total=True))
                            reduced_chi2.append(chi/(len(outer_frequency) - len(initial_vars)))
                    # Second iteration model i.e. [__, one, __, two, __, three]
                    else:
                        if self.flags["verbose"]:
                            print(f"{model_idx + 1}: {model_dictionary[model_idx]} harvey model w/ white noise fixed")

                        # Get the correct number of initial variables as the current Harvey model
                        delta = 2 * (self.num_laws - (model_idx//2 + 1))
                        initial_vars = list(harvey_parameters[:(-delta-1)])
                        initial_vars.append(harvey_parameters[-1])
                        # Try to fit Harvey function with bounds
                        try:
                            fitted_vars, _cv = curve_fit(
                                self.functions[model_idx//2 + 1],
                                self.binned_frequency,
                                self.binned_power,
                                p0=initial_vars,
                                sigma=self.binned_error,
                                bounds=bounds_list[model_idx//2]
                            )
                        except RuntimeError:
                            fitted_parameters_list.append([])
                            reduced_chi2.append(np.inf)
                        else:
                            fitted_parameters_list.append(fitted_vars)
                            chi, _p = chisquare(f_obs=outer_power, f_exp=harvey(outer_frequency, fitted_vars, total=True))
                            reduced_chi2.append(chi/(len(outer_frequency) - len(initial_vars) + 1))

                # Fitted curve successfully
                if np.isfinite(min(reduced_chi2)):
                    model = reduced_chi2.index(min(reduced_chi2))
                    # [(1, 2) -> 1, (3, 4) -> 2, (5, 6) -> 3]
                    self.num_laws = ((model)//2) + 1
                    if self.flags["verbose"]:
                        print(f"Based on reduced chi-squared statistic: model {model+1}")

                    # Replace parameters with the best model's fitted parameters
                    self.fitted_harvey_parameters = fitted_parameters_list[model]
                    best_bounds = bounds_list[self.num_laws - 1]
                    final_pars = np.zeros((self.num_mc_iter, 2*self.num_laws + 12))

                    # Lower bound is 1.0
                    sm_par = max(4.0*(self.numax / NUMAX_SUN)**0.2, 1.0)
                    # Process has succeeded so don't run again
                    again = False

                # Failed curve fit
                else:
                    print("Failed to fit Harvey model...")
                    again = True

            # Second iteration or later
            else:
                # Try to fit Harvey function
                try:
                    # pdb.set_trace()
                    harvey_parameters, _cv = curve_fit(
                        self.functions[self.num_laws],
                        self.binned_frequency,
                        self.binned_power,
                        p0=harvey_parameters,
                        sigma=self.binned_error,
                        bounds=best_bounds
                    )
                # Failed to fit Harvey function
                except RuntimeError as e:
                    print("Failed to fit Harvey function...")
                    again = True
                except ValueError as e:
                    print(f"Updating bounds... {e}")
                    again = True
                    best_bounds[0][-1] -= 0.01
                    best_bounds[1][-1] += 0.01
                # Successfully fit Harvey function
                else:
                    again = False

            # Retry if curve fitting failed
            if again:
                print("Retrying...")
                continue

            final_pars[mc_iteration, 0 : 2*self.num_laws + 1] = harvey_parameters

            fwhm = sm_par * self.dnu / self.resolution
            # Standard deviation
            sig = fwhm/np.sqrt(8 * np.log(2))
            gauss_kernel = Gaussian1DKernel(int(sig))
            # Smooth power spectrum before background correction
            pssm = convolve_fft(random_power, gauss_kernel)

            # Model of stellar background
            background_model = harvey(self.frequency, harvey_parameters, total=True)

            # Correct for edge effects and residual slope in Gaussian fit
            region_frequency = np.copy(self.frequency[self.mask])

            # The difference between the first point of the power envelope and the last point of the power envelope
            delta_y = pssm[self.mask][-1] - pssm[self.mask][0]
            # The width of the power envelope
            delta_x = region_frequency[-1] - region_frequency[0]
            # The general slope of the power envelope
            slope = delta_y / delta_x
            # y-intercept
            y_intercept = (-1.0 * slope * region_frequency[0]) + pssm[self.mask][0]
            corrected = np.array([slope * region_frequency[z] + y_intercept for z in range(len(region_frequency))])
            corrected_pssm = [pssm[self.mask][z] - corrected[z] + background_model[self.mask][z] for z in range(len(pssm[self.mask]))]

            plot_x = np.array(list(self.frequency[self.mask]) + list(self.frequency[~self.mask]))
            ss = np.argsort(plot_x)
            plot_x = plot_x[ss]
            pssm = np.array(corrected_pssm + list(background_model[~self.mask]))
            pssm = pssm[ss]
            
            # Subtract stellar background
            pssm_bgcorr = pssm - background_model

            region_power = pssm_bgcorr[self.mask]
            # Index of maximum power i.e. numax estimation
            idx = self.return_max(region_power, index=True)
            final_pars[mc_iteration, 2*self.num_laws + 1] = region_frequency[idx]
            final_pars[mc_iteration, 2*self.num_laws + 2] = region_power[idx]

            if list(region_frequency) != []:
                numax_gaussian_bounds = self.gaussian_bounds(region_frequency, region_power)
                initial_vars = [
                    0.0,
                    max(region_power),
                    region_frequency[idx],
                    (max(region_frequency) - min(region_frequency))/8.0/np.sqrt(8.0*np.log(2.0))
                ]
                while True:
                    try:
                        # Fit Gaussian to the background corrected power envelope to find numax
                        numax_gaussian_parameters, _cv = curve_fit(
                            gaussian,
                            region_frequency,
                            region_power,
                            p0=initial_vars,
                            # bounds=numax_gaussian_bounds
                        )
                        break
                    except ValueError:
                        print("Updating numax Gaussian's parameters bounds...")
                        numax_gaussian_bounds[0][0] -= 0.1
                        numax_gaussian_bounds[0][1] -= 0.1
                        numax_gaussian_bounds[0][2] -= 0.1
                        numax_gaussian_bounds[1][0] += 0.1
                        numax_gaussian_bounds[1][1] += 0.1
                        numax_gaussian_bounds[1][2] += 0.1
                offset, amplitude, center, width = numax_gaussian_parameters
                final_pars[mc_iteration, 2*self.num_laws + 3] = center
                final_pars[mc_iteration, 2*self.num_laws + 4] = amplitude
                final_pars[mc_iteration, 2*self.num_laws + 5] = width

            # Background correction
            self.bg_corr = random_power / background_model

            # Optional smoothing of PS to remove fine structure TODO: This doesn't seem to be initialised anywhere
            if False and self.smooth_ps is not None:
                boxkernel = Box1DKernel(int(np.ceil(self.smooth_ps / self.resolution)))
                bg_corr_smooth = convolve(self.bg_corr, boxkernel)
            else:
                bg_corr_smooth = np.array(self.bg_corr)

            self.width = WIDTH_SUN * (width / NUMAX_SUN)
            self.numax = center
            self.sm_par = 4.0 * (self.numax / NUMAX_SUN)**0.2
            self.dnu = 0.22 * (self.numax**0.797)
            self.num_dnus = self.width / self.dnu

            # Frequencies in the power envelope
            psd = bg_corr_smooth[self.mask]
            lag, auto = self.corr(region_frequency, psd)
            lag_peaks, auto_peaks = self.max_elements(lag, auto)

            # Pick the peak closest to the modelled numax
            idx = self.return_max(lag_peaks, index=True, dnu=True)
            best_lag = lag_peaks[idx]
            best_auto = auto_peaks[idx]
            lag_peaks[idx] = np.nan
            auto_peaks[idx] = np.nan

            dnu_gaussian_bounds = self.gaussian_bounds(lag, auto, best_x=best_lag, sigma=10**(-2.0))
            initial_vars = [
                np.mean(auto),
                best_auto,
                best_lag,
                0.01 * 2.0 * best_lag
            ]
            # Fitting Gaussian to ACF to find deltanu
            dnu_gaussian_parameters, _cv = curve_fit(
                gaussian,
                lag,
                auto,
                p0=initial_vars,
                bounds=dnu_gaussian_bounds
            )
            
            offset, amplitude, center, width = dnu_gaussian_parameters

            zoom_mask = (lag >= best_lag - 3.0*width) & (lag <= best_lag + 3.0*width)
            zoom_lag = lag[zoom_mask]
            zoom_auto = auto[zoom_mask]
            dnu_fit = gaussian(zoom_lag, offset, amplitude, center, width)
            idx = self.return_max(dnu_fit, index=True)

            # Force deltanu measurement to be the initial guess
            if self.flags["force"]:
                self.dnu = self.guess
            # Use deltanu found using ACF
            else:
                self.dnu = zoom_lag[idx]
            
            self.get_ridges(mc_iteration)
            final_pars[mc_iteration, 2*self.num_laws + 6] = self.dnu

            if mc_iteration == 0 and not again:
                self.pssm = pssm
                self.region_frequency = region_frequency
                self.region_power = region_power
                self.numax_gaussian_parameters = numax_gaussian_parameters
                self.psd = psd
                self.lag = lag
                self.auto = auto
                self.best_lag = best_lag
                self.best_auto = best_auto
                self.lag_peaks = lag_peaks
                self.auto_peaks = auto_peaks
                self.zoom_lag = zoom_lag
                self.zoom_auto = zoom_auto
                self.dnu_gaussian_parameters = dnu_gaussian_parameters
                self.plot_fitbg()

            mc_iteration += 1

        if self.num_mc_iter > 1:
            if self.flags["verbose"]:
                print(f"numax (smoothed): {final_pars[0, 2*self.num_laws + 1]:.2f} +/- {mad_std(final_pars[:, 2*self.num_laws + 1]):.2f} muHz")
                print(f"maxamp (smoothed): {final_pars[0, 2*self.num_laws + 2]:.2f} +/- {mad_std(final_pars[:, 2*self.num_laws + 2]):.2f} ppm^2/muHz")
                print(f"numax (gaussian): {final_pars[0, 2*self.num_laws + 3]:.2f} +/- {mad_std(final_pars[:, 2*self.num_laws + 3]):.2f} muHz")
                print(f"maxamp (gaussian): {final_pars[0, 2*self.num_laws + 4]:.2f} +/- {mad_std(final_pars[:, 2*self.num_laws + 4]):.2f} ppm^2/muHz")
                print(f"fwhm (gaussian): {final_pars[0, 2*self.num_laws + 5]:.2f} +/- {mad_std(final_pars[:, 2*self.num_laws + 5]):.2f} muHz")
                print(f"dnu: {final_pars[0, 2*self.num_laws + 6]:.2f} +/- {mad_std(final_pars[:, 2*self.num_laws + 6]):.2f} muHz")
                print("-------------------------------------------------\n")
            
            results.append(
                [
                    self.target,
                    final_pars[0, 2*self.num_laws + 1],
                    mad_std(final_pars[:, 2*self.num_laws + 1]),
                    final_pars[0, 2*self.num_laws + 2],
                    mad_std(final_pars[:, 2*self.num_laws + 2]),
                    final_pars[0, 2*self.num_laws + 3],
                    mad_std(final_pars[:, 2*self.num_laws + 3]),
                    final_pars[0, 2*self.num_laws + 4],
                    mad_std(final_pars[:, 2*self.num_laws + 4]),
                    final_pars[0, 2*self.num_laws + 5],
                    mad_std(final_pars[:, 2*self.num_laws + 5]),
                    final_pars[0, 2*self.num_laws + 6],
                    mad_std(final_pars[:, 2*self.num_laws + 6])
                ]
            )

            self.plot_mc(final_pars)

        else:
            if again:
                results.append([self.target, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            else:
                if self.flags["verbose"]:
                    print(f"numax (smoothed): {final_pars[0, 2*self.num_laws + 1]:.2f} muHz")
                    print(f"maxamp (smoothed): {final_pars[0, 2*self.num_laws + 2]:.2f} ppm^2/muHz")
                    print(f"numax (gaussian): {final_pars[0, 2*self.num_laws + 3]:.2f} muHz")
                    print(f"maxamp (gaussian): {final_pars[0, 2*self.num_laws + 4]:.2f} ppm^2/muHz")
                    print(f"fwhm (gaussian): {final_pars[0, 2*self.num_laws + 5]:.2f} muHz")
                    print(f"dnu: {final_pars[0, 2*self.num_laws + 6]:.2f} muHz")
                    print("-------------------------------------------------\n")

                results.append(
                    [
                        self.target,
                        final_pars[0, 2*self.num_laws + 1],
                        0.0,
                        final_pars[0, 2*self.num_laws + 2],
                        0.0,
                        final_pars[0, 2*self.num_laws + 3],
                        0.0,
                        final_pars[0, 2*self.num_laws + 4],
                        0.0,
                        final_pars[0, 2*self.num_laws + 5],
                        0.0,
                        final_pars[0, 2*self.num_laws + 6],
                        0.0
                    ]
                )

        self.write_bgfit(results[0])
    
    def check(self) -> bool:
        """Check if a prior numax has been provided already by star info or has been estimated with findex.
        
        Returns
        result : bool
            true if a numax value exists for the target otherwise false
        """

        # SYD needs some prior knowledge about numax to work well
        # (either from findex module or from star info csv)
        if self.numax is None:
            print("WARNING: Suggested use of this pipeline requires either")
            print("stellar properties to estimate a numax or run the entire")
            print("pipeline from scratch (i.e. find_excess) first to")
            print("statistically determine a starting point for numax.")

            return False
        else:
            return True

    def get_white_noise(self, random_power: np.ndarray) -> np.ndarray:
        """Get white noise.
        
        Parameters
        ----------
        random_power : np.ndarray
            randomised power spectrum

        Returns
        -------
        noise : np.ndarray
            white noise sampled depending on the power spectrum's Nyquist frequency
        """

        # Sample noise
        if self.nyquist < 400.0:
            mask = (self.frequency > 200.0) & (self.frequency < 270.0)
            noise = np.mean(random_power[mask])
        elif self.nyquist > 400.0 and self.nyquist < 5000.0:
            mask = (self.frequency > 4000.0) & (self.frequency < 4167.0)
            noise = np.mean(random_power[mask])
        elif self.nyquist > 5000.0 and self.nyquist < 9000.0:
            mask = (self.frequency > 8000.0) & (self.frequency < 8200.0)
            noise = np.mean(random_power[mask])
        else:
            mask = (self.frequency > (0.9 * max(self.frequency))) & (self.frequency < max(self.frequency))
            noise = np.mean(random_power[mask])
        return noise

    def plot_fitbg(self) -> None:
        """Plots result of fitbg routine."""

        fig = plt.figure(figsize=(12, 12))

        # Time series data
        ax1 = fig.add_subplot(3, 3, 1)
        ax1.plot(self.time, self.flux, "w-")
        ax1.set_xlim([min(self.time), max(self.time)])
        ax1.set_title(r"$\rm Time \,\, series$")
        ax1.set_xlabel(r"$\rm Time \,\, [days]$")
        ax1.set_ylabel(r"$\rm Flux$")

        # Initial background guesses
        ax2 = fig.add_subplot(3, 3, 2)
        ax2.plot(self.frequency[self.frequency < self.envelope_edges[0]], self.power[self.frequency < self.envelope_edges[0]], "w-", zorder=0)
        ax2.plot(self.frequency[self.frequency > self.envelope_edges[1]], self.power[self.frequency > self.envelope_edges[1]], "w-", zorder=0)
        ax2.plot(self.frequency[self.frequency < self.envelope_edges[0]], self.smooth_power[self.frequency < self.envelope_edges[0]], "r-", linewidth=0.75, zorder=1)
        ax2.plot(self.frequency[self.frequency > self.envelope_edges[1]], self.smooth_power[self.frequency > self.envelope_edges[1]], "r-", linewidth=0.75, zorder=1)

        for r in range(self.num_laws):
            ax2.plot(self.frequency, harvey(self.frequency, [self.fitted_harvey_parameters[2*r], self.fitted_harvey_parameters[2*r + 1], self.fitted_harvey_parameters[-1]]), color="blue", linestyle=":", linewidth=1.5, zorder=3)
        ax2.plot(self.frequency, harvey(self.frequency, self.fitted_harvey_parameters, total=True), color="blue", linewidth=2.0, zorder=4)
        ax2.errorbar(self.binned_frequency, self.binned_power, yerr=self.binned_error, color="lime", markersize=0.0, fillstyle="none", ls="None", marker="D", capsize=3, ecolor="lime", elinewidth=1, capthick=2, zorder=2)
        for m, n in zip(self.mnu_original, self.a_original):
            ax2.plot(m, n, color="blue", fillstyle="none", mew=3.0, marker="s", markersize=5.0)
        ax2.axvline(self.envelope_edges[0], color="darkorange", linestyle="dashed", linewidth=2.0, zorder=1, dashes=(5, 5))
        ax2.axvline(self.envelope_edges[1], color="darkorange", linestyle="dashed", linewidth=2.0, zorder=1, dashes=(5, 5))
        ax2.axhline(self.noise, color="blue", linestyle="dashed", linewidth=1.5, zorder=3, dashes=(5, 5))
        ax2.set_xlim([min(self.frequency), max(self.frequency)])
        ax2.set_ylim([min(self.power), 1.25*max(self.power)])
        ax2.set_title(r"$\rm Initial \,\, guesses$")
        ax2.set_xlabel(r"$\rm Frequency \,\, [\mu Hz]$")
        ax2.set_ylabel(r"$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$")
        ax2.set_xscale("log")
        ax2.set_yscale("log")

        # Fitted background
        ax3 = fig.add_subplot(3, 3, 3)
        ax3.plot(self.frequency[self.frequency < self.envelope_edges[0]], self.power[self.frequency < self.envelope_edges[0]], "w-", linewidth=0.75, zorder=0)
        ax3.plot(self.frequency[self.frequency > self.envelope_edges[1]], self.power[self.frequency > self.envelope_edges[1]], "w-", linewidth=0.75, zorder=0)
        ax3.plot(self.frequency[self.frequency < self.envelope_edges[0]], self.smooth_power[self.frequency < self.envelope_edges[0]], "r-", linewidth=0.75, zorder=1)
        ax3.plot(self.frequency[self.frequency > self.envelope_edges[1]], self.smooth_power[self.frequency > self.envelope_edges[1]], "r-", linewidth=0.75, zorder=1)
        for r in range(self.num_laws):
            ax3.plot(self.frequency, harvey(self.frequency, [self.fitted_harvey_parameters[2*r], self.fitted_harvey_parameters[2*r + 1], self.fitted_harvey_parameters[-1]]), color="blue", linestyle=":", linewidth=1.5, zorder=3)
        ax3.plot(self.frequency, harvey(self.frequency, self.fitted_harvey_parameters, total=True), color="blue", linewidth=2.0, zorder=4)
        ax3.errorbar(self.binned_frequency, self.binned_power, yerr=self.binned_error, color="lime", markersize=0.0, fillstyle="none", ls="None", marker="D", capsize=3, ecolor="lime", elinewidth=1, capthick=2, zorder=2)
        ax3.axvline(self.envelope_edges[0], color="darkorange", linestyle="dashed", linewidth=2.0, zorder=1, dashes=(5, 5))
        ax3.axvline(self.envelope_edges[1], color="darkorange", linestyle="dashed", linewidth=2.0, zorder=1, dashes=(5, 5))
        ax3.axhline(self.noise, color="blue", linestyle="dashed", linewidth=1.5, zorder=3, dashes=(5, 5))
        ax3.plot(self.frequency, self.pssm, color="yellow", linewidth=2.0, linestyle="dashed", zorder=5)
        ax3.set_xlim([min(self.frequency), max(self.frequency)])
        ax3.set_ylim([min(self.power), max(self.power)*1.25])
        ax3.set_title(r"$\rm Fitted \,\, model$")
        ax3.set_xlabel(r"$\rm Frequency \,\, [\mu Hz]$")
        ax3.set_ylabel(r"$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$")
        ax3.set_xscale("log")
        ax3.set_yscale("log")

        # Smoothed power excess w/ Gaussian
        ax4 = fig.add_subplot(3, 3, 4)
        ax4.plot(self.region_frequency, self.region_power, "w-", zorder=0)
        idx = self.return_max(self.region_power, index=True)
        ax4.plot([self.region_frequency[idx]], [self.region_power[idx]], color="red", marker="s", markersize=7.5, zorder=0)
        ax4.axvline([self.region_frequency[idx]], color="white", linestyle="--", linewidth=1.5, zorder=0)
        gaus = gaussian(self.region_frequency, self.numax_gaussian_parameters[0], self.numax_gaussian_parameters[1], self.numax_gaussian_parameters[2], self.numax_gaussian_parameters[3])
        plot_min = 0.0
        if min(self.region_power) < plot_min:
            plot_min = min(self.region_power)
        if min(gaus) < plot_min:
            plot_min = min(gaus)
        plot_max = 0.0
        if max(self.region_power) > plot_max:
            plot_max = max(self.region_power)
        if max(gaus) > plot_max:
            plot_max = max(gaus)
        plot_range = plot_max-plot_min
        ax4.plot(self.region_frequency, gaus, "b-", zorder=3)
        ax4.axvline([self.numax_gaussian_parameters[2]], color="blue", linestyle=":", linewidth=1.5, zorder=2)
        ax4.plot([self.numax_gaussian_parameters[2]], [self.numax_gaussian_parameters[1]], color="b", marker="D", markersize=7.5, zorder=1)
        ax4.set_title(r"$\rm Smoothed \,\, bg$-$\rm corrected \,\, PS$")
        ax4.set_xlabel(r"$\rm Frequency \,\, [\mu Hz]$")
        ax4.set_xlim([min(self.region_frequency), max(self.region_frequency)])
        ax4.set_ylim([plot_min - 0.1*plot_range, plot_max + 0.1*plot_range])

        # Background-corrected PS with n highest peaks
        peaks_f, peaks_p = self.max_elements(self.region_frequency, self.psd)
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.plot(self.region_frequency, self.psd, "w-", zorder=0, linewidth=1.0)
        ax5.scatter(peaks_f, peaks_p, s=25.0, edgecolor="r", marker="s", facecolor="none", linewidths=1.0)
        ax5.set_title(r"$\rm Bg$-$\rm corrected \,\, PS$")
        ax5.set_xlabel(r"$\rm Frequency \,\, [\mu Hz]$")
        ax5.set_ylabel(r"$\rm Power$")
        ax5.set_xlim([min(self.region_frequency), max(self.region_frequency)])
        ax5.set_ylim([min(self.psd) - 0.025*(max(self.psd) - min(self.psd)), max(self.psd) + 0.1*(max(self.psd) - min(self.psd))])

        # ACF for determining dnu
        ax6 = fig.add_subplot(3, 3, 6)
        ax6.plot(self.lag, self.auto, "w-", zorder=0, linewidth=1.0)
        ax6.scatter(self.lag_peaks, self.auto_peaks, s=30.0, edgecolor="r", marker="^", facecolor="none", linewidths=1.0)
        ax6.axvline(self.best_lag, color="white", linestyle="--", linewidth=1.5, zorder=2)
        ax6.scatter(self.best_lag, self.best_auto, s=45.0, edgecolor="lime", marker="s", facecolor="none", linewidths=1.0)
        ax6.plot(self.zoom_lag, self.zoom_auto, "r-", zorder=5, linewidth=1.0)
        ax6.set_title(r"$\rm ACF \,\, for \,\, determining \,\, \Delta\nu$")
        ax6.set_xlabel(r"$\rm Frequency \,\, separation \,\, [\mu Hz]$")
        ax6.set_xlim([min(self.lag), max(self.lag)])
        ax6.set_ylim([min(self.auto) - 0.05*(max(self.auto) - min(self.auto)), max(self.auto) + 0.1*(max(self.auto) - min(self.auto))])

        # dnu fit
        fit = gaussian(self.zoom_lag, self.dnu_gaussian_parameters[0], self.dnu_gaussian_parameters[1], self.dnu_gaussian_parameters[2], self.dnu_gaussian_parameters[3])
        idx = self.return_max(fit, index=True)
        plot_lower = min(self.zoom_auto)
        if min(fit) < plot_lower:
            plot_lower = min(fit)
        plot_upper = max(self.zoom_auto)
        if max(fit) > plot_upper:
            plot_upper = max(fit)
        ax7 = fig.add_subplot(3, 3, 7)
        ax7.plot(self.zoom_lag, self.zoom_auto, "w-", zorder=0, linewidth=1.0)
        ax7.axvline(self.best_lag, color="red", linestyle="--", linewidth=1.5, zorder=2)
        ax7.plot(self.zoom_lag, fit, color="lime", linewidth=1.5)
        ax7.axvline([self.zoom_lag[idx]], color="lime", linestyle="--", linewidth=1.5)
        ax7.set_title(r"$\rm \Delta\nu \,\, fit$")
        ax7.set_xlabel(r"$\rm Frequency \,\, separation \,\, [\mu Hz]$")
        ax7.annotate(r"$\Delta\nu = %.2f$"%self.zoom_lag[idx], xy=(0.025, 0.85), xycoords="axes fraction", fontsize=18, color="lime")
        ax7.set_xlim([min(self.zoom_lag), max(self.zoom_lag)])
        ax7.set_ylim([plot_lower - 0.05*(plot_upper - plot_lower), plot_upper + 0.1*(plot_upper - plot_lower)])

        # echelle diagram
        print("Plotting LogNorm echelle...")
        ax8 = fig.add_subplot(3, 3, 8)
        ax8.imshow(
            np.ma.masked_invalid(self.echelle),
            extent=self.extent,
            interpolation="bilinear",
            aspect="auto",
            origin="lower",
            cmap="jet",
            norm=LogNorm(
                vmin=np.nanmedian(self.echelle_copy),
                vmax=np.nanmax(self.echelle_copy)
                )
            )
        print("Finished plotting LogNorm echelle...")
        ax8.axvline([self.dnu], color="white", linestyle="--", linewidth=1.0, dashes=(5, 5))
        ax8.set_title(r"$\rm \grave{E}chelle \,\, diagram$")
        ax8.set_xlabel(r"$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$" % self.dnu)
        ax8.set_ylabel(r"$\rm \nu \,\, [\mu Hz]$")
        ax8.set_xlim([0.0, 2.0*self.dnu])
        ax8.set_ylim([self.envelope_edges[0], self.envelope_edges[1]])

        ax9 = fig.add_subplot(3, 3, 9)
        ax9.plot(self.mod_frequency, self.collapsed_power, color="white", linestyle="-", linewidth=0.75)
        ax9.set_title(r"$\rm Collapsed \,\, \grave{e}chelle \,\, diagram$")
        ax9.set_xlabel(r"$\rm \nu \,\, mod \,\, %.2f \,\, [\mu Hz]$"%self.dnu)
        ax9.set_ylabel(r"$\rm Collapsed \,\, power$")
        ax9.set_xlim([0.0, 2.0*self.dnu])
        ax9.set_ylim([min(self.collapsed_power) - 0.025*(max(self.collapsed_power) - min(self.collapsed_power)), max(self.collapsed_power) + 0.05*(max(self.collapsed_power) - min(self.collapsed_power))])

        plt.tight_layout()
        # Save plot
        if self.flags["save"]:
            plt.savefig(f"{self.path}{self.target}_fitbg.png", dpi=300)
        if self.flags["show_plots"]:
            plt.show() 
        plt.close()

    def plot_mc(self, final_pars):
        """
        Plots Markov-Chain/Monte-Carlo simulation.
        """
    
        plt.figure(figsize=(12, 8))

        titles = [
            r"$\rm Smoothed \,\, \nu_{max} \,\, [\mu Hz]$",
            r"$\rm Smoothed \,\, A_{max} \,\, [ppm^{2} \mu Hz^{-1}]$",
            r"$\rm Gaussian \,\, \nu_{max} \,\, [\mu Hz]$",
            r"$\rm Gaussian \,\, A_{max} \,\, [ppm^{2} \mu Hz^{-1}]$",
            r"$\rm Gaussian \,\, FWHM \,\, [\mu Hz]$",
            r"$\rm \Delta\nu \,\, [\mu Hz]$"
            ]

        for i in range(6):
            ax = plt.subplot(2, 3, i+1)
            ax.hist(final_pars[:, 2*self.num_laws + (i+1)], color="cyan", histtype="step", lw=2.5, facecolor="0.75")
            ax.set_title(titles[i])

        plt.tight_layout()
        # Save plot
        if self.flags["save"]:
            plt.savefig(f"{self.path}{self.target}_mc.png", dpi=300)
        if self.flags["show_plots"]:
            plt.show()
        plt.close()
        
    def return_max(self, array, index=False, dnu=False):
        """Returns the min/max of the given array or its index.
        
        Parameters
        ----------
        array : np.ndarray
            array
        index : bool
            will return the index if true
        dnu : bool
            will find the minimum of the |array - estimated_dnu|
        
        Returns
        result : int | float
            either the min/max value or the index of the min/max value
        """
        if dnu:
            exp_dnu = 0.22*(self.numax**0.797)
            lst = list(np.absolute(np.copy(array)-exp_dnu))
            idx = lst.index(min(lst))
        else:
            lst = list(array)
            idx = lst.index(max(lst))
        if index:
            return idx
        else:
            return lst[idx]

    def corr(self, _frequency: np.ndarray, power: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates autocorrelation.
        
        Parameters
        ----------
        frequency : np.ndarray
            power spectrum frequency
        power : np.ndarray
            power spectrum power
        
        Returns
        -------
        lag : np.ndarray
            frequency lag
        auto : np.ndarray
            autocorrelation
        """

        lag = np.arange(0.0, len(power)) * self.resolution
        auto = np.real(np.fft.fft(np.fft.ifft(power)*np.conj(np.fft.ifft(power))))

        lower_limit = self.dnu / 4.0
        upper_limit = 2.0*self.dnu + self.dnu/4.0

        mask = np.ma.getmask(np.ma.masked_inside(lag, lower_limit, upper_limit))
        lag = lag[mask]
        auto = auto[mask]
        auto -= min(auto)
        auto /= max(auto)
    
        return lag, auto

    def max_elements(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the first n peaks of y and its corresponding x co-ordinate.
        
        Parameters
        ----------
        x : np.ndarray
            x data
        y : np.ndarray
            y data
        
        Returns
        -------
        peaks_x : np.ndarray
            the x co-ordinates of the first n peaks of y
        peaks_y : np.ndarray
            the y co-ordinates of the first n peaks of y
        """

        s = np.argsort(y)
        peaks_y = y[s][-int(self.n_peaks):][::-1]
        peaks_x = x[s][-int(self.n_peaks):][::-1]

        return peaks_x, peaks_y

    def gaussian_bounds(self, x, y, best_x=None, sigma=None) -> Tuple:
        """TODO: Check what this does
        
        Parameters
        ----------
        x : np.ndarray
            x data
        y : np.ndarray
            y data
        best_x : float
            TODO: Don't know
        sigma : float
            TODO: Don't know
        
        Returns
        -------
        result : tuple
            Gaussian bounds
        """

        if sigma is None:
            sigma = (max(x) - min(x))/8.0/np.sqrt(8.0 * np.log(2.0))

        b = np.zeros((2, 4)).tolist()
        b[1][0] = np.inf
        # TODO: This seems to imply that int(np.max()) will be None which would make the first assignment invalid
        b[1][1] = 2.0*np.max(y)
        if not int(np.max(y)):
            b[1][1] = np.inf
        if best_x is not None:
            b[0][2] = 0.999 * best_x
            b[1][2] = 1.001 * best_x
        else:
            b[0][2] = np.min(x)
            b[1][2] = np.max(x)
        b[0][3] = sigma
        b[1][3] = np.max(x) - np.min(x)
        return tuple(b)

    def get_ridges(self, mc_iteration: int, start: float = 0.0):
        """
        TODO: Not sure what this does exactly
        """

        if mc_iteration == 0:
            self.dnu = self.get_best_dnu()

        # Calculate echelle
        echelle, _gridx, _gridy, extent = self.calculate_echelle()
        # pdb.set_trace()
        N, M = echelle.shape[0], echelle.shape[1]
        echelle_copy = np.array(list(echelle.reshape(-1)))

        n = int(np.ceil(self.dnu / self.resolution))
        xax = np.zeros(n)
        yax = np.zeros(n)
        modx = self.frequency % self.dnu

        for k in range(n):
            use = np.where((modx >= start) & (modx < start + self.resolution))[0]
            if len(use) == 0:
                continue
            xax[k] = np.median(modx[use])
            yax[k] = np.sum(self.bg_corr[use])
            start += self.resolution

        xax = np.array(list(xax) + list(xax + self.dnu))
        yax = np.array(list(yax) + list(yax)) - min(yax)

        if self.flags["clip"]:
            if self.clip_value != 0.0:
                cut = self.clip_value
            else:
                cut = (np.nanmax(echelle_copy) + np.nanmedian(echelle_copy)) / 2.0
            cut_mask = ~np.isnan(echelle_copy)
            cut_mask[cut_mask] &= echelle_copy[cut_mask] > cut
            echelle_copy[cut_mask] = cut

        if mc_iteration == 0:
            self.echelle_copy = echelle_copy
            self.echelle = echelle_copy.reshape((N, M))
            self.extent = extent
            self.mod_frequency = xax
            self.collapsed_power = yax

    def get_best_dnu(self) -> float:
        """Estimates deltanu."""

        # Create range around current deltanu estimate
        dnus = np.arange(0.95*self.dnu, 1.05*self.dnu, 0.01)
        difference = np.zeros_like(dnus)

        for dnu_idx, current_dnu in enumerate(dnus):
            n = int(np.ceil(current_dnu / self.resolution))
            xax = np.zeros(n)
            yax = np.zeros(n)
            modx = self.frequency % current_dnu

            start = 0.0
            for k in range(n):
                use = np.where((modx >= start) & (modx < start + self.resolution))[0]
                if len(use) == 0:
                    continue
                xax[k] = np.median(modx[use])
                yax[k] = np.sum(self.bg_corr[use])
                start += self.resolution

            difference[dnu_idx] = np.max(yax) - np.mean(yax)

        idx = self.return_max(difference, index=True)
        return dnus[idx]

    def calculate_echelle(self, nox=20, startx=0.0):
        """
        Calculates echelle.
        """

        # Smooths power spectra before calculating echelle TODO: Currently does nothing
        if self.flags["echelle_smooth"]:
            boxkernel = Box1DKernel(int(np.ceil(self.echelle_filter / self.resolution)))
            _smooth_y = convolve(self.bg_corr, boxkernel)

        noy = int(np.ceil((max(self.frequency) - min(self.frequency))/self.dnu))

        if nox > 2 and noy > 5:
            xax = np.arange(0.0, self.dnu + (self.dnu/nox)/2.0, self.dnu/nox)
            yax = np.arange(min(self.frequency), max(self.frequency), self.dnu)
            # pdb.set_trace()
            arr = np.zeros((len(xax), len(yax)))
            gridx = np.zeros(len(xax))
            gridy = np.zeros(len(yax))

            modx = self.frequency % self.dnu
            starty = min(self.frequency)

            for ii in range(len(gridx)):
                for jj in range(len(gridy)):
                    use = np.where((modx >= startx) & (modx < startx + self.dnu/nox) & (self.frequency >= starty) & (self.frequency < starty + self.dnu))[0]
                    if len(use) == 0:
                        arr[ii, jj] = np.nan
                    else:
                        arr[ii, jj] = np.sum(self.bg_corr[use])
                    gridy[jj] = starty + self.dnu/2.0
                    starty += self.dnu
                gridx[ii] = startx + self.dnu/nox/2.0
                starty = min(self.frequency)
                startx += self.dnu/nox
            smoothed = arr
            dim = smoothed.shape

            smoothed_2 = np.zeros((2*dim[0], dim[1]))
            smoothed_2[0:dim[0], :] = smoothed
            smoothed_2[dim[0]:(2*dim[0]), :] = smoothed
            smoothed = np.swapaxes(smoothed_2, 0, 1)
            extent = [min(gridx) - self.dnu/nox/2.0, 2*max(gridx) + self.dnu/nox/2.0, min(gridy) - self.dnu/2.0, max(gridy) + self.dnu/2.0]

            return smoothed, np.array(list(gridx) + list(gridx + self.dnu)), gridy, extent
        else:
            return None
    
    def write_bgfit(self, results):
        """
        Writes result of fitbg to output file.
        """

        variables = ["target", "numax(smooth)", "numax(smooth)_err", "maxamp(smooth)", "maxamp(smooth)_err", "numax(gauss)", "numax(gauss)_err", "maxamp(gauss)", "maxamp(gauss)_err", "fwhm", "fwhm_err", "dnu", "dnu_err"]
        ascii.write(np.array(results), f"{self.path}{self.target}_globalpars.csv", names=variables, delimiter=",", overwrite=True)
