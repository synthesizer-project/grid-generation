"""
Download M05 and convert to HDF5 synthesizer grid.
"""

import os
import tarfile
from pathlib import Path

import numpy as np
import wget
from synthesizer.conversions import flam_to_fnu
from unyt import Hz, angstrom, cm, dimensionless, erg, s, yr
from utils import get_model_filename

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser


def download_data(input_dir, url):
    """
    TODO: These could be replaced by our own mirror
    """

    # Download the original data to the working directory
    filename = wget.download(url)

    Path(input_dir).mkdir(parents=True, exist_ok=True)

    # Untar the main directory
    tar = tarfile.open(filename)
    tar.extractall(path=input_dir)
    tar.close()
    os.remove(filename)


def make_grid(model, imf, hr_morphology, input_dir, grid_dir):
    """Main function to convert BC03 grids and
    produce grids used by synthesizer"""

    # get synthesizer model name
    synthesizer_model_name = get_model_filename(model)

    print(synthesizer_model_name)

    # Define output
    out_filename = f"{grid_dir}/{synthesizer_model_name}.hdf5"

    # NOTE THE LOWEST METALLICITY MODEL DOES NOT HAVE YOUNG AGES so don't use
    metallicities = np.array(
        [0.001, 0.01, 0.02, 0.04]
    )  # array of avialable metallicities

    # Codes for converting metallicty
    metallicity_codes = {
        0.0001: "10m4",
        0.001: "0001",
        0.01: "001",
        0.02: "002",
        0.04: "004",
        0.07: "007",
    }

    # Open first file to get age
    fn = (
        f"{input_dir}/sed.{imf}z{metallicity_codes[metallicities[0]]}."
        f"{hr_morphology}"
    )
    ages_, _, lam_, flam_ = np.loadtxt(fn).T

    ages_Gyr = np.sort(np.array(list(set(ages_))))  # Gyr
    ages = ages_Gyr * 1e9  # yr

    lam = lam_[ages_ == ages_[0]] * angstrom

    spec = np.zeros((len(ages), len(metallicities), len(lam)))

    for imetal, metallicity in enumerate(metallicities):
        for ia, age_Gyr in enumerate(ages_Gyr):
            fn = (
                f"{input_dir}/sed.{imf}z{metallicity_codes[metallicity]}."
                f"{hr_morphology}"
            )
            print(imetal, ia, fn)
            ages_, _, lam_, flam_ = np.loadtxt(fn).T

            flam = flam_[ages_ == age_Gyr]
            fnu = flam_to_fnu(lam, flam * erg / s / angstrom / cm**2)
            spec[ia, imetal] = fnu

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
    parser = Parser(description="Maraston download and grid creation")

    # Unpack the arguments
    args = parser.parse_args()

    # the directory to store downloaded input files
    input_dir = args.input_dir

    # the directory to store the grid
    grid_dir = args.grid_dir

    sps_name = "maraston05"

    # Define the model metadata
    model_name = sps_name
    imfs = ["ss"]  # , 'kr'
    model = {
        "sps_name": sps_name,
        "sps_version": False,
        "alpha": False,
    }

    # append sps_name to input_dir to define where to store downloaded input
    # files
    input_dir += f"/{sps_name}"

    # create directory to store downloaded output if it doesn't exist
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    # Define the download URL
    original_data_url = {}
    original_data_url["ss"] = """
http://www.icg.port.ac.uk/~maraston/SSPn/SED/Sed_Mar05_SSP_Salpeter.tar.gz"""

    for imf in imfs:
        # Download the data if necessary
        if args.download:
            print(original_data_url[imf])
            download_data(input_dir, original_data_url[imf])

        if imf == "ss":
            model["imf_type"] = "bpl"
            model["imf_masses"] = [0.1, 100]
            model["imf_slopes"] = [2.35]

        for hr_morphology in ["rhb"]:
            model["sps_variant"] = hr_morphology

            # Get and write the grid
            make_grid(model, imf, hr_morphology, input_dir, grid_dir)
