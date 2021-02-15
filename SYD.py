"""
The SYD PYpline -> asteroseismic SYD pipeline (Huber et al. 2009) originally written in IDL
translated into python for all your automated asteroseismic needs.
"""

import argparse
import glob
import multiprocessing as mp
import os
import subprocess
import sys
import time
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import *
from find_excess import FindExcess
from fit_background import FitBackground
from functions import *
from target_data import TargetData

# Defaults
DEFAULT_STAR_INFO = "Files/star_info.csv"
DEFAULT_TODO = "Files/todo.txt"
DEFAULT_DATA = "Files/data"


def main(args: list) -> None:
    """Main function that assigns tasks to run given by the list of IDs listed in todo.txt.

    Parameters
    ----------
    args : list
        command line arguments
    """

    args = _parse_args(args)

    power_spectrum = PowerSpectrum(
        args.mission,
        args.smoothing_freq,
        args.findex,
        args.fitbg,
        args.verbose,
        args.show,
        args.ignore,
        args.todo,
        args.data,
        args.star_info
    )

    # Multithread
    if args.parallel:
        power_spectrum.flags["ignore"] = False
        num_threads = args.num_threads if args.num_threads is not None else mp.cpu_count() - 1

        tasks = []
        # Split tasks between threads
        for n in range(num_threads):
            if list(power_spectrum.targets[n::num_threads]) != []:
                tasks.append(list(power_spectrum.targets[n::num_threads]))
        # Assign tasks to the thread pool
        with mp.Pool(processes=len(tasks)) as pool:
            pool.map(power_spectrum.assign_tasks, tasks)
    # Single thread
    else:
        power_spectrum.assign_tasks(power_spectrum.targets)

    if power_spectrum.flags["verbose"]:
        print("Combining results into single csv file.\n")

    # Concatenate data together
    subprocess.call(["python", "scrape_output.py"], shell=True)


# Argument parsing
def _parse_args(args: list) -> argparse.Namespace:
    """Parses command line arguments.

    Parameters
    ----------
    args : list
        unparsed command line arguments

    Returns
    -------
    args : argparse.Namespace
        parsed command line arguments
    """

    parser = argparse.ArgumentParser(
        description="This is a script to run the new version of the SYD PYpeline."
    )

    # Optional arguments
    parser.add_argument(
        "-e", "-x", "--e", "--x", "-ex", "--ex", "-excess", "--excess",
        help="Use this to turn the find excess function off",
        dest="findex",
        action="store_false"
    )
    parser.add_argument(
        "-b", "--b", "-bg", "--bg", "-background", "--background",
        help="Use this to disable the background fitting process (although not highly recommended)",
        dest="fitbg",
        action="store_false"
    )
    parser.add_argument(
        "-f", "--f", "-fr", "--fr", "-smooth", "--smooth",
        help="Frequency for smoothing power spectrum in muHz. Default=1 muHz.",
        default=1.0,
        dest="smoothing_freq"
    )
    parser.add_argument(
        "-m", "--m", "-ms", "--ms", "-mission", "--mission",
        help="Mission used to collect data. Options: Kepler, TESS, K2. Default is Kepler.",
        choices=["Kepler", "TESS", "K2"], default="TESS", dest="mission"
    )
    parser.add_argument(
        "-v", "--v", "-verbose", "--verbose",
        help="Turn on verbose",
        dest="verbose",
        action="store_false"
    )
    parser.add_argument(
        "-s", "--s", "-show", "--show",
        help="Show plots",
        dest="show",
        action="store_false"
    )
    parser.add_argument(
        "-i", "--i", "-ignore", "--ignore",
        help="Ignore multiple target output suppression",
        dest="ignore",
        action="store_true"
    )
    parser.add_argument(
        "-p", "--p", "-parallel", "--parallel",
        help="Enables parallel processing",
        dest="parallel",
        action="store_true"
    )
    parser.add_argument(
        "-t", "--t", "-threads", "--threads",
        help="Specify the number of threads to use during parallel processing",
        dest="num_threads",
        nargs="?",
        const=mp.cpu_count()-1,
        default=None,
        type=int
    )
    parser.add_argument(
        "-todo", "--todo",
        help="File path to target list",
        dest="todo",
        nargs="?",
        const=DEFAULT_TODO,
        default=DEFAULT_TODO,
        type=lambda x: _is_valid_file(parser, x)
    )
    parser.add_argument(
        "-data", "--data",
        help="File path to data folder",
        dest="data",
        nargs="?",
        const=DEFAULT_DATA,
        default=DEFAULT_DATA,
        type=lambda x: _is_valid_folder(parser, x)
    )
    parser.add_argument(
        "-info", "--info",
        help="File path to star info",
        dest="star_info",
        nargs="?",
        const=DEFAULT_STAR_INFO,
        default=DEFAULT_STAR_INFO,
        type=str
    )

    return parser.parse_args(args)


def _is_valid_file(parser, path):
    if not os.path.isfile(path):
        parser.error(f"The file at the path {path} does not exist!")
    else:
        return path


def _is_valid_folder(parser, path):
    if not os.path.isdir(path):
        parser.error(f"The directory at the path {path} does not exist!")
    else:
        return path


# Utility functions
def _get_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads light curve and power spectrum data files.

    Parameters
    ----------
    path : str
        file path of either light curve or power spectrum data

    Returns
    -------
    x : np.ndarray
        time in case of light curves and frequency in case of power spectra
    y : np.ndarray
        flux in case of light curves and power in case of power spectra
    """

    f = open(path, "r")
    lines = f.readlines()
    f.close()

    x = np.array([float(line.strip().split()[0]) for line in lines])
    y = np.array([float(line.strip().split()[1]) for line in lines])

    return x, y


##########################################################################################
#                                                                                        #
#                                      DICTIONARIES                                      #
#                                                                                        #
##########################################################################################


class PowerSpectrum:
    # TODO: Finish PowerSpectrum docstring
    """Main class that handles the runtime of the SYD-PYpline.

    Attributes
    ----------
    mission : str
        mission used to collect data. Options: Kepler, TESS, K2.
    smoothing_freq : float
        frequency for smoothing power spectrum in muHz.
    flags : dict
        A collection of flags that concern the runtime of the pipeline.
    """

    def __init__(
            self,
            mission: str,
            smoothing_freq: float,
            findex: bool,
            fitbg: bool,
            verbose: bool,
            show_plots: bool,
            ignore: bool,
            todo: str,
            data: str,
            star_info: str
    ) -> None:
        """Initialises the class PowerSpectrum.

        Parameters
        ----------
        mission : str
            mission used to collect data. Options: Kepler, TESS, K2
        smoothing_freq : float
            frequency for smoothing power spectrum in muHz
        findex : bool
            flag to run the find excess routine
        fitbg : bool
            flag to run the fit background routine
        verbose : bool
            flag to increase verbosity
        show_plots : bool
            flag to show plots
        ignore : bool
            flag to ignore multiple target output suppression
        todo : str
            file path to target list
        star_info : str
            file path to star info
        """
        self.mission = mission
        self.smoothing_freq = smoothing_freq
        self.flags = {
            "findex": findex,
            "fitbg": fitbg,
            "verbose": verbose,
            "show_plots": show_plots,
            "ignore": ignore
        }
        self.targets = {}
        self.data_folder = data if os.path.isdir(data) else DEFAULT_DATA
        self.get_info(todo, star_info)
        self.set_plot_params()

    def get_info(self, todo, star_info):
        """Loads target list, constants, routine parameters and star info.

        Parameters
        ----------
        todo : str
            file path to target list
        star_info : str
            file path to star info
        """

        # Load list of targets
        with open(todo, "r") as f:
            target_list = np.array([line.strip().split()[0] for line in f.readlines()])

        # Turn off verbosity and plots in case of multiple targets
        if len(target_list) > 1 and not self.flags["ignore"]:
            self.flags["verbose"] = False
            self.flags["show_plots"] = False

        # Set file paths for each target
        for target in target_list:
            self.targets[target] = {}
            self.targets[target]["path"] = "/".join(self.data_folder.split("/")[:-1]) + f"/results/{target}/"

        # Initialise parameters needed for the find excess routine
        if self.flags["findex"]:
            self.findex = FindExcess(
                self.targets,
                verbose=self.flags["verbose"],
                show_plots=self.flags["show_plots"],
                smoothing_frequency=self.smoothing_freq
            )
        # Initialise parameters needed for the fit background routine
        if self.flags["fitbg"]:
            self.fitbg = FitBackground(
                verbose=self.flags["verbose"],
                show_plots=self.flags["show_plots"],
                smoothing_frequency=self.smoothing_freq
            )

        # Load star info
        self.get_star_info(star_info)

    def set_plot_params(self) -> None:
        """Sets plot styling and parameters."""

        plt.style.use("dark_background")
        plt.rcParams.update(
            {
                "agg.path.chunksize": 10000,
                "mathtext.fontset": "stix",
                "figure.autolayout": True,
                "lines.linewidth": 1,
                "axes.titlesize": 18.0,
                "axes.labelsize": 16.0,
                "axes.linewidth": 1.25,
                "axes.formatter.useoffset": False,
                "xtick.major.size": 10.0,
                "xtick.minor.size": 5.0,
                "xtick.major.width": 1.25,
                "xtick.minor.width": 1.25,
                "xtick.direction": "inout",
                "ytick.major.size": 10.0,
                "ytick.minor.size": 5.0,
                "ytick.major.width": 1.25,
                "ytick.minor.width": 1.25,
                "ytick.direction": "inout",
            }
        )

    def assign_tasks(self, targets: list) -> None:
        """Processes targets in target list.

        Parameters
        ----------
        targets : list
            list of targets to process
        """

        completion_times = []

        for target in targets:
            target_data = self.load_data(target)
            if target_data is not None:
                # Given star info data
                numax = self.targets[target]["numax"]
                dnu = self.targets[target]["dnu"]
                # Default snr TODO: Change default value?
                snr = 2
                print(f"Processing target {target}...")
                if self.flags["findex"]:
                    print("Finding excess...")
                    # Load target data into the find excess routine
                    self.findex.update_target(target_data)
                    # Override previous guesses with the find excess routine's results
                    numax, snr, dnu = self.findex.find_excess()
                if self.flags["fitbg"]:
                    print("Fitting background...")
                    # Load target data, numax, snr and dnu into the find excess routine
                    self.fitbg.update_target(target_data, numax, snr, dnu)
                    start = time.time()
                    completed = self.fitbg.fit_background()
                    elapsed = time.time() - start
                    completion_times.append(elapsed)
        avg_time = np.mean(np.array(completion_times))
        print(f"Average completion time is {avg_time//60} minutes and {avg_time%60:.2f} seconds.")

##########################################################################################
#                                                                                        #
#                               READING/WRITING TO/FROM FILES                            #
#                                                                                        #
##########################################################################################

    def get_star_info(self, star_info: str, cols: Iterable[str] = ["rad", "logg", "teff"]) -> None:
        """Loads information about target stars as given in target file.

        Parameters
        ----------
        star_info : str
            file path to star info
        cols : Iterable[str]
            list of columns to store info from
        """

        # Load star info
        if os.path.exists(star_info):
            df = pd.read_csv(star_info)
            target_ids = df.targets.values.tolist()
            star_info_columns = df.columns.values.tolist()

            for current_id in self.targets:
                if int(current_id) in target_ids:
                    idx = target_ids.index(int(current_id))
                    # Add specified information
                    for col in cols:
                        self.targets[current_id][col] = df.loc[idx, col]
                    # Mass, numax and deltanu estimates
                    if "numax" in star_info_columns:
                        self.targets[current_id]["mass"] = estimate_mass(
                            self.targets[current_id]["rad"],
                            self.targets[current_id]["logg"]
                        )
                        self.targets[current_id]["numax"] = df.loc[idx, "numax"]
                        self.targets[current_id]["dnu"] = estimate_delta_nu_from_numax(df.loc[idx, "numax"])
                    else:
                        self.targets[current_id]["mass"] = estimate_mass(
                            self.targets[current_id]["rad"],
                            self.targets[current_id]["logg"]
                        )
                        self.targets[current_id]["numax"] = estimate_numax(
                            self.targets[current_id]["rad"],
                            self.targets[current_id]["mass"],
                            self.targets[current_id]["teff"]
                        )
                        self.targets[current_id]["dnu"] = estimate_deltanu(
                            self.targets[current_id]["rad"],
                            self.targets[current_id]["mass"]
                        )

    def remove_artefact(
            self,
            lc_time: np.ndarray,
            ps_frequency: np.ndarray,
            ps_power: np.ndarray,
            resolution: float
    ) -> np.ndarray:
        """Removes SC artefacts in Kepler power spectra by replacing them with noise (using linear interpolation)
        following an exponential distribution; known artefacts are:

        1) 1./LC harmonics
        2) unknown artefacts at high frequencies (>5000 muHz)
        3) excess power between 250-400 muHz (in Q0 and Q3 data only??)

        Parameters
        ----------
        lc_time : np.ndarray
            light curve times
        ps_frequency : np.ndarray
            power spectrum frequencies
        ps_power : np.ndarray
            power spectrum power
        resolution : float
            resolution

        Returns
        -------
        power : np.ndarray
            power where artefact frequencies are filled with noise
        """

        f, a = ps_frequency, ps_power

        # LC period in Msec
        lc = 29.4244 * 60 * 1e-6
        lcp = 1.0/lc
        art = (1.0 + np.arange(14)) * lcp

        # Lower limit of artefact
        un1 = [4530.0, 5011.0, 5097.0, 5575.0, 7020.0, 7440.0, 7864.0]
        # Upper limit of artefact
        un2 = [4534.0, 5020.0, 5099.0, 5585.0, 7030.0, 7450.0, 7867.0]

        usenoise = np.where((f >= max(f) - 100.0) & (f <= max(f) - 50.0))[0]
        # Noise floor
        noisefl = np.mean(a[usenoise])

        # Routine 1: Remove 1/LC artefacts by subtracting +/- 5 muHz given each artefact
        for i in range(0, len(art)):
            if art[i] < np.max(f):

                use = np.where((f > art[i] - 5.0*resolution) & (f < art[i] + 5.0*resolution))[0]
                if use[0] != -1:
                    a[use] = noisefl * np.random.chisquare(2, len(use))/2.0

        # Routine 2: Remove artefacts as identified in un1 & un2
        for i in range(0, len(un1)):
            if un1[i] < np.max(f):
                use = np.where((f > un1[i]) & (f < un2[i]))[0]
                if use[0] != -1:
                    a[use] = noisefl * np.random.chisquare(2, len(use))/2.0

        # Routine 3: Remove two wider artefacts as identified in un1 & un2
        un1 = [240.0, 500.0]
        un2 = [380.0, 530.0]

        for i in range(0, len(un1)):
            # un1[i] : frequency where artefact starts
            # un2[i] : frequency where artefact ends
            # flower : initial frequency to start fitting routine (aka un1[i] - 20)
            # fupper : final frequency to end fitting routine (aka un2[i] + 20)
            flower, fupper = un1[i] - 20, un2[i] + 20
            usenoise = np.where(
                ((f >= flower) & (f <= un1[i])) |
                ((f >= un2[i]) & (f <= fupper))
            )[0]
            # Coefficients for linear fit
            m, b = np.polyfit(f[usenoise], a[usenoise], 1)
            # Index of artefact frequencies (ie. 240-380 muHz)
            use = np.where((f >= un1[i]) & (f <= un2[i]))[0]
            # Fill artefact frequencies with noise
            a[use] = (f[use]*m + b) * np.random.chisquare(2, len(use))/2.0
        # Set new amplitude
        return a

    def load_data(self, target: str) -> TargetData:
        """Loads light curve and power spectrum data of the current target.

        Parameters
        ----------
        target : str
            current target ID
        """

        # Now done at beginning to make sure it only does this one per target
        if glob.glob(f"{self.data_folder}/{target}_*") != []:
            # Check if light curve files exist
            if not os.path.isfile(f"{self.data_folder}/{target}_LC.txt"):
                if self.flags["verbose"]:
                    print(f"Error: {self.data_folder}/{target}_LC.txt not found")
                return None
            # Check if power spectrum files exist
            if not os.path.isfile(f"{self.data_folder}/{target}_PS.txt"):
                if self.flags["verbose"]:
                    print(f"Error: {self.data_folder}/{target}_PS.txt not found")
                return None

            # Load light curve
            lc_time, lc_flux = _get_file(f"{self.data_folder}/{target}_LC.txt")
            # Calculate cadence in seconds
            cadence = int(np.nanmedian(np.diff(lc_time) * 24.0 * 60.0 * 60.0))
            # Calculate Nyquist frequency in muHz
            nyquist = 10**6 / (2.0*cadence)
            if self.flags["verbose"]:
                print(f"# LIGHT CURVE: {len(lc_time)} lines of data read")

            # load power spectrum
            ps_frequency, ps_power = _get_file(f"{self.data_folder}/{target}_PS.txt")
            # Calculate oversample factor and resolution
            oversample = int(round((1.0/((max(lc_time) - min(lc_time))*0.0864))/(ps_frequency[1] - ps_frequency[0])))
            resolution = (ps_frequency[1] - ps_frequency[0]) * oversample

            # Remove artefacts for Kepler targets
            if self.mission == "Kepler":
                # ps_power = self.remove_artefact(lc_time, ps_frequency, ps_power, resolution)
                # if self.flags["verbose"]:
                #     print(f"Removing artefacts from {self.mission} data.")
                print("Unsupported currently...")
            if self.flags["verbose"]:
                print(f"# POWER SPECTRUM: {len(ps_frequency)} lines of data read")

            if self.flags["verbose"]:
                print("-------------------------------------------------")
                print(f"Target: {target}")
                if oversample == 1:
                    print("critically sampled")
                else:
                    print(f"oversampled by a factor of {oversample}")
                print(f"time series cadence: {cadence} seconds")
                print(f"power spectrum resolution: {resolution:.6f} muHz")
                print("-------------------------------------------------")

            return TargetData(
                target,
                self.targets[target]["path"],
                lc_time,
                lc_flux,
                cadence,
                nyquist,
                ps_frequency,
                ps_power,
                oversample,
                resolution
            )
        else:
            print(f"Error: data not found for target {target}")
            return None

##########################################################################################
#                                                                                        #
#                                        INITIATE                                        #
#                                                                                        #
##########################################################################################


if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed//60} minutes and {elapsed%60:.2f} seconds!")
