"""
Download Maraston2011 and convert to HDF5 synthesizer grid.
"""

import os
import tarfile
from pathlib import Path

import numpy as np
import wget
from synthesizer.conversions import llam_to_lnu
from unyt import Angstrom, Hz, dimensionless, erg, s, yr
from utils import get_model_filename

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser


def download_data(
    output_dir,
    data_url="http://www.icg.port.ac.uk/~maraston/M11/SSP_M11_Pickles.tar.gz",
):
    """
    Download Maraston+11 data
    Args:
        input_dir (string):
            directory to download and unpack data into
        data_url (string):
            URL from which to fetch the data
    Returns:
        None
    """
    filename = wget.download(
        data_url
    )  # download the original data to the working directory

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("filename:", filename)
    # --- untar main directory
    tar = tarfile.open(filename)
    tar.extractall(path=output_dir)
    tar.close()
    os.remove(filename)


def make_grid(model, imf, variant, output_dir, grid_dir):
    """Main function to convert Maraston 2011 and
    produce grids used by synthesizer
    Args:
        model (dict):
            dictionary containing model parameters
        imf (string):
            Initial mass function, can be one of Salpeter,
            Kroupa or Chabrier
        variant (string):
            Model variant to use
        output_dir (string):
            directory where the raw Maraston+11 files are read from
        grid_dir (string):
            directory where the the grid is created
    Returns:
        fname (string):
            output filename
    """

    synthesizer_model_name = get_model_filename(model)
    print(synthesizer_model_name)

    # output filename
    out_filename = f"{grid_dir}/{synthesizer_model_name}.hdf5"

    metallicities = np.array([0.02])  # array of available metallicities

    metallicity_codes = {0.02: "002"}  # codes for converting metallicty

    # define the extension
    if variant:
        extension = "_" + variant
    else:
        extension = ""

    fn = (
        f"{output_dir}/ssp_M11_Pickles{extension}.{imf_code[imf]}"
        f"z{metallicity_codes[metallicities[0]]}"
    )

    ages_, _, lam_, llam_ = np.loadtxt(fn).T  # llam is in (ergs /s /AA /Msun)

    ages_Gyr = np.sort(np.array(list(set(ages_))))  # Gyr
    ages = ages_Gyr * 1e9

    lam = lam_[ages_ == ages_[0]] * Angstrom

    spec = np.zeros((len(ages), len(metallicities), len(lam)))

    # at each point in spec convert the units
    for imetal, metallicity in enumerate(metallicities):
        for ia, age_Gyr in enumerate(ages_Gyr):
            print(imetal, ia, fn)
            ages_, _, lam_, llam_ = np.loadtxt(fn).T

            llam = llam_[ages_ == age_Gyr] * erg / s / Angstrom
            lnu = llam_to_lnu(lam, llam)
            spec[ia, imetal] = lnu

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename)

    # A dictionary with Boolean values for each axis, where True
    # indicates that the attribute should be interpolated in
    # logarithmic space.
    log_on_read = {"ages": True, "metallicities": False}

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=lam,
        spectra={"incident": spec * erg / s / Hz},
        alt_axes=("ages", "metallicities"),
        log_on_read=log_on_read,
        weight="initial_masses",
    )

    # Include the specific ionising photon luminosity
    out_grid.add_specific_ionising_lum()


# Lets include a way to call this script not via an entry point
if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="Maraston+11 download and grid creation")

    # Unpack the arguments
    args = parser.parse_args()

    # the directory to store downloaded input files
    input_dir = args.input_dir

    # the directory to store the grid
    grid_dir = args.grid_dir

    # Define the model metadata
    sps_name = "maraston11"
    imf_code = {"salpeter": "ss", "kroupa": "kr", "chabrier": "cha"}
    default_model = {
        "sps_name": sps_name,
        "sps_variant": False,
        "sps_version": False,
        "alpha": False,
        "imf_masses": [0.1, 100],
    }

    # The location to untar the original data
    output_dir = f"{input_dir}/{sps_name}"

    # Download the data if necessary
    if args.download:
        download_data(output_dir)

    for variant in [
        False,
        "nearIRextended",
        "UVtheoretical",
        "UVtheoretical_nearIRextended",
    ]:
        if (variant is False) or (variant == "nearIRextended"):
            imfs = ["salpeter", "kroupa", "chabrier"]

        if (variant == "UVtheoretical") or (
            variant == "UVtheoretical_nearIRextended"
        ):
            imfs = ["salpeter"]

        for imf in imfs:
            print(variant)

            model = default_model | {"imf_type": imf, "sps_variant": variant}

            # Get and write the grid
            make_grid(model, imf, variant, output_dir, grid_dir)
