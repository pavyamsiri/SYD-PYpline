"""
The find excess routine that automatically finds power excess due to solar-like oscillations
using a frequency resolved collapsed autocorrelation function.
"""

import os
from target_data import TargetData
from typing import Iterable, Tuple, Dict

from astropy.io import ascii
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

from functions import *
from constants import *
from target_data import TargetData

##########################################################################################
#                                                                                        #
#                            [CRUDE] FIND POWER EXCESS ROUTINE                           #
#                                                                                        #
##########################################################################################
# TODOS
# 1) add in process to check the binning/crude bg fit and retry if desired
# 2) allow user to pick model instead of it picking the highest SNR
# 3) check if the Gaussian is fully resolved
# 4) maybe change the part that guesses the offset (mean of entire frequency range - not just the beginning)
# ADDED
# 1) Ability to add more trials for numax determination


class FindExcess:
    """Main class that handles the execution of the find excess routine."""

    def __init__(
            self,
            save: bool = True,
            verbose: bool = False,
            show_plots: bool = False,
            step: float = 0.25,
            bin_width: float = 0.005,
            smoothing_frequency: float = 1.0,
            num_trials: int = 3,
            lower: float = 10.0,
            upper: float = 4000.0
    ) -> None:
        """Initialises parameters needed for the find excess routine.

        Parameters
        ----------
        targets : dict
            list of targets to run the find excess routine on
        save : bool
            flag to save results
        verbose : bool
            flag to increase verbosity
        show_plots : bool
            flag to show plots
        step : float
            TODO: Not sure what this means
        bin_width : float
            logarithmic binning width
        smoothing_frequency : float
            frequency for smoothing power spectrum in muHz
        num_trials : int
            number of trials
        lower : float
            lower frequency bound
        upper : float
            upper frequency bound
        """

        self.flags: Dict[str, bool] = {
            "save": save,
            "verbose": verbose,
            "show_plots": show_plots
        }
        self.step: float = step
        self.bin_width: float = bin_width
        self.smoothing_frequency: float = smoothing_frequency
        self.lower: float = lower
        self.upper: float = upper
        self.num_trials: int = num_trials

        # Number of rows of ACF trial plots
        total_plots = self.num_trials + 3
        if total_plots % 3 == 0:
            self.num_rows: int = (total_plots - 1)//3
        else:
            self.num_rows: int = total_plots//3

        # Uninitialised target data
        self.target: str = None
        self.path: str = None
        # Time series
        self.time: np.ndarray = None
        self.flux: np.ndarray = None
        self.resolution: float = None
        self.short_cadence: bool = None
        # Power spectrum
        self.frequency: np.ndarray = None
        self.power: np.ndarray = None
        # Binned power spectrum
        self.binned_frequency: np.ndarray = None
        self.binned_power: np.ndarray = None
        # Interpolated power from fitted spline
        self.interpolated_power: np.ndarray = None
        # Background corrected power
        self.bgcorr_power: np.ndarray = None
        # ACF trials
        self.mean_frequencies: Iterable[np.ndarray] = None  # Used to be md. Not sure what it means
        self.cumulative_sums: Iterable[np.ndarray] = None
        self.fitted_frequencies: Iterable[np.ndarray] = None
        self.fitted_power: Iterable[np.ndarray] = None
        self.fitted_numax: Iterable[float] = None
        self.fitted_gauss: Iterable[float] = None
        self.fitted_snr: Iterable[float] = None

    def update_target(
            self,
            target_data: TargetData
    ) -> None:
        """Updates current target.

        Parameters
        ----------
        target_data : TargetData
            file, time series and power spectrum data of the new target
        """

        # Initialise target data
        self.target = target_data.target
        self.path = target_data.path

        # Time series
        self.time = np.copy(target_data.lc_time)
        self.flux = np.copy(target_data.lc_flux)
        self.short_cadence = target_data.cadence/60.0 < 10.0

        # Power spectrum
        mask = np.ones_like(target_data.ps_frequency, dtype=bool)
        # Lower frequency bound
        if self.lower is not None:
            mask *= np.ma.getmask(
                np.ma.masked_greater_equal(target_data.ps_frequency, self.lower)
            )
        # Upper frequency bound
        if self.upper is not None:
            mask *= np.ma.getmask(
                np.ma.masked_less_equal(target_data.ps_frequency, self.upper)
            )
        # Nyquist bound
        else:
            mask *= np.ma.getmask(np.ma.masked_less_equal(target_data.ps_frequency, target_data.nyquist))
        self.frequency = np.copy(target_data.ps_frequency[mask])
        self.power = np.copy(target_data.ps_power[mask])
        self.resolution = target_data.resolution/target_data.oversample

        # Clear arrays
        self.binned_frequency = None
        self.binned_power = None
        self.interpolated_power = None
        self.bgcorr_power = None
        # Reinitialise lists
        self.mean_frequencies = []
        self.cumulative_sums = []
        self.fitted_frequencies = []
        self.fitted_power = []
        self.fitted_numax = []
        self.fitted_gauss = []
        self.fitted_snr = []

        # Make save folder
        if self.flags["save"]:
            if not os.path.isdir(self.path):
                os.makedirs(self.path)

    def find_excess(self) -> Tuple[float, float, float]:
        """Automatically finds power excess due to solar-like oscillations using a frequency resolved
        collapsed autocorrelation function.

        Returns
        -------
        measured_numax : float
            the measured numax of the highest S/N ACF trial
        measured_snr : float
            the S/N of the best fitting ACF trial
        measured_dnu : float
            the measured dnu of the highest S/N ACF trial calculated from the measured numax
        """

        if self.bin_width is not None:
            self.binned_frequency, self.binned_power = bin_data(self.frequency, self.power, self.bin_width)
            if self.flags["verbose"]:
                print(f"binned to {len(self.binned_frequency)} datapoints")

            # Smooth power spectrum by convolving with a box kernel
            box_size = int(np.ceil(self.smoothing_frequency/self.resolution))
            smooth_power = convolve(self.binned_power, Box1DKernel(box_size))
            smooth_frequency = self.binned_frequency[int(box_size/2):-int(box_size/2)]
            smooth_power = smooth_power[int(box_size/2):-int(box_size/2)]

            # Interpolate spline function over smoothed power spectrum
            spl = InterpolatedUnivariateSpline(smooth_frequency, smooth_power, k=1)
            self.interpolated_power = spl(self.frequency)
            self.bgcorr_power = self.power/self.interpolated_power

            if not self.short_cadence:
                boxes = np.logspace(np.log10(0.5), np.log10(25.0), self.num_trials)
            else:
                boxes = np.logspace(np.log10(50.0), np.log10(500.0), self.num_trials)

            results = []

            # Divide power spectrum into subsets
            for idx, box in enumerate(boxes):
                subset = np.ceil(box / self.resolution)
                steps = np.ceil((box * self.step)/self.resolution)

                cumulative_sum = np.zeros_like(self.frequency)
                mean_frequency = np.zeros_like(self.frequency)
                jj = 0
                start = 0

                while (start + subset) <= len(self.frequency):
                    frequency = self.frequency[int(start):int(start + subset)]
                    power = self.bgcorr_power[int(start):int(start + subset)]

                    _lag = np.arange(0.0, len(power)) * self.resolution
                    # Auto-correlation
                    auto = np.real(np.fft.fft(np.fft.ifft(power)*np.conj(np.fft.ifft(power))))
                    corr = np.absolute(auto - np.mean(auto))

                    cumulative_sum[jj] = np.sum(corr)
                    mean_frequency[jj] = np.mean(frequency)

                    start += steps
                    jj += 1

                # Add mean frequencies where its correlation is non-zero
                self.mean_frequencies.append(
                    mean_frequency[~np.ma.getmask(np.ma.masked_values(cumulative_sum, 0.0))]
                )

                # Non-zero cumulative correlation
                cumulative_sum = cumulative_sum[~np.ma.getmask(np.ma.masked_values(cumulative_sum, 0.0))] \
                    - min(cumulative_sum[~np.ma.getmask(np.ma.masked_values(cumulative_sum, 0.0))])
                cumulative_sum = list(cumulative_sum/max(cumulative_sum))

                self.cumulative_sums.append(np.array(cumulative_sum))
                # Fitted numax
                max_idx = cumulative_sum.index(max(cumulative_sum))
                self.fitted_numax.append(self.mean_frequencies[idx][max_idx])

                # Attempt to fit Gaussian to collapsed ACF
                try:
                    fitted_vars, _covar = curve_fit(
                        gaussian,
                        self.mean_frequencies[idx],
                        self.cumulative_sums[idx],
                        p0=[
                            np.mean(self.cumulative_sums[idx]),
                            1.0 - np.mean(self.cumulative_sums[idx]),
                            self.mean_frequencies[idx][max_idx],
                            WIDTH_SUN * (self.mean_frequencies[idx][max_idx]/NUMAX_SUN)
                        ]
                    )
                    offset, _amplitude, center, width = fitted_vars
                # Failed to fit Gaussian
                except RuntimeError as _e:
                    results.append([self.target, np.nan, np.nan, -np.inf])
                else:
                    self.fitted_frequencies.append(np.linspace(min(mean_frequency), max(mean_frequency), 10000))
                    self.fitted_power.append(
                        gaussian(
                            self.fitted_frequencies[idx],
                            fitted_vars[0],
                            fitted_vars[1],
                            fitted_vars[2],
                            fitted_vars[3]
                        )
                    )

                    # Max power divided by offset
                    snr = max(self.fitted_power[idx]) / offset
                    # Bound snr by 100
                    snr = min(snr, 100.0)

                    self.fitted_snr.append(snr)
                    # Store estimated numax from Gaussian fit
                    self.fitted_gauss.append(center)
                    # Results: [target, numax, dnu, snr]
                    results.append([self.target, center, estimate_delta_nu_from_numax(center), snr])

                    if self.flags["verbose"]:
                        print(f"power excess trial {idx + 1}: numax = {center:.2f} +/- {(np.absolute(width)/2.0):.2f}")
                        print(f"S/N: {snr:.2f}")

            snr_comparisons = [entry[-1] for entry in results]
            best_model = snr_comparisons.index(max(snr_comparisons))
            if self.flags["verbose"]:
                print(f"picking model {best_model + 1}")
            self.write_excess(results[best_model])

            measured_numax = results[best_model][1]
            measured_dnu = results[best_model][2]
            measured_snr = results[best_model][3]

            # Plot results
            self.plot_findex()

            return measured_numax, measured_snr, measured_dnu

    def write_excess(self, results: list) -> None:
        """Writes result of find excess routine to output file.

        Parameters
        ----------
        results : list
            results from find excess routine
        """

        variables = ["target", "numax", "dnu", "snr"]
        ascii.write(
            np.array(results),
            f"{self.path}{self.target}_findex.csv",
            names=variables,
            delimiter=",",
            overwrite=True
        )

    def plot_findex(self):
        """Plots the results of the find excess routine."""

        # Create new figure
        plt.figure(figsize=(12, 8))

        # Top left: light curve
        ax1 = plt.subplot(1 + self.num_rows, 3, 1)
        ax1.plot(self.time, self.flux, "w-")
        ax1.set_xlim([min(self.time), max(self.time)])
        ax1.set_title(r"$\rm Time \,\, series$")
        ax1.set_xlabel(r"$\rm Time \,\, [days]$")
        ax1.set_ylabel(r"$\rm Flux$")

        # Top middle: log-log power spectrum with crude background fit
        ax2 = plt.subplot(1 + self.num_rows, 3, 2)
        # Uncorrected power spectrum
        ax2.loglog(self.frequency, self.power, "w-")
        # TODO: Why 100.0 muHz and not lower?
        ax2.set_xlim([100.0, max(self.frequency)])
        ax2.set_ylim([min(self.power), 1.25 * max(self.power)])
        ax2.set_title(r"$\rm Crude \,\, background \,\, fit$")
        ax2.set_xlabel(r"$\rm Frequency \,\, [\mu Hz]$")
        ax2.set_ylabel(r"$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$")
        # Binned smooth power spectrum
        if self.bin_width is not None:
            ax2.loglog(self.binned_frequency, self.binned_power, "r-")
        # Spline interpolated power
        ax2.loglog(self.frequency, self.interpolated_power, color="lime", linestyle="-", lw=2.0)

        # Top right: Crude background-corrected power spectrum
        ax3 = plt.subplot(1 + self.num_rows, 3, 3)
        ax3.plot(self.frequency, self.bgcorr_power, "w-")
        # TODO: Why 100.0 muHz and not lower?
        ax3.set_xlim([100.0, max(self.frequency)])
        ax3.set_ylim([0.0, 1.25 * max(self.bgcorr_power)])
        ax3.set_title(r"$\rm Background \,\, corrected \,\, PS$")
        ax3.set_xlabel(r"$\rm Frequency \,\, [\mu Hz]$")
        ax3.set_ylabel(r"$\rm Power \,\, [ppm^{2} \mu Hz^{-1}]$")

        # ACF trials to determine numax
        for idx in range(self.num_trials):
            xran = max(self.fitted_frequencies[idx]) - min(self.fitted_frequencies[idx])

            ymax = max(
                max(self.cumulative_sums[idx]),
                max(self.fitted_power[idx])
            )

            yran = np.absolute(ymax)

            ax = plt.subplot(1 + self.num_rows, 3, 4 + idx)
            ax.plot(self.mean_frequencies[idx], self.cumulative_sums[idx], "w-")
            ax.axvline(self.fitted_numax[idx], linestyle="dotted", color="r", linewidth=0.75)
            ax.set_title(r"$\rm Collapsed \,\, ACF \,\, [trial \,\, %d]$" % (idx + 1))
            ax.set_xlabel(r"$\rm Frequency \,\, [\mu Hz]$")
            ax.set_ylabel(r"$\rm Arbitrary \,\, units$")
            ax.plot(
                self.fitted_frequencies[idx],
                self.fitted_power[idx],
                color="lime",
                linestyle="-",
                linewidth=1.5
            )
            ax.axvline(self.fitted_gauss[idx], color="lime", linestyle="--", linewidth=0.75)
            ax.set_xlim([min(self.fitted_frequencies[idx]), max(self.fitted_frequencies[idx])])
            ax.set_ylim([-0.05, ymax + 0.15*yran])
            ax.annotate(
                r"$\rm SNR=%3.2f$" % self.fitted_snr[idx],
                xy=(
                    min(self.fitted_frequencies[idx]) + 0.05*xran,
                    ymax + 0.025*yran
                ),
                fontsize=18
            )

        plt.tight_layout()
        # Save plot
        if self.flags["save"]:
            plt.savefig(f"{self.path}{self.target}_findex.png", dpi=300)
        if self.flags["show_plots"]:
            plt.show()
        plt.close()
