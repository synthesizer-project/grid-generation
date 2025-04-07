"""
Download the Maraston2013 SPS model and convert to HDF5 synthesizer grid.
"""

import os

import numpy as np
from synthesizer.conversions import llam_to_lnu
from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser
from unyt import Angstrom, Hz, dimensionless, erg, s, yr
from utils import get_model_filename


def make_grid(synthesizer_model_name, imf, input_dir, grid_dir):
    """Main function to convert Maraston 2013 and
    produce grids used by synthesizer
    Args:
        synthesizer_model_name (string):
            name of the model to be saved.
        imf (string):
            Initial mass function, Salpeter or
            Kroupa
        input_dir (string):
            directory where the raw Maraston+13 files are read from
        grid_dir (string):
            directory where the grids are created.
    Returns:
        fname (string):
            output filename
    """

    # define output
    out_filename = f"{grid_dir}/{synthesizer_model_name}.hdf5"

    metallicities = np.array(
        [0.001, 0.01, 0.02, 0.04]
    )  # array of available metallicities

    metallicity_codes = {
        0.001: "0001",
        0.01: "001",
        0.02: "002",
        0.04: "004",
    }  # codes for converting metallicty

    # open first raw data file to get age
    fn = (
        f"{input_dir}/sed_M13.{imf_code[imf]}"
        f"z{metallicity_codes[metallicities[0]]}"
    )

    ages_, _, lam_, llam_ = np.loadtxt(fn).T  # llam is in (ergs /s /AA /Msun)

    ages_Gyr = np.sort(np.array(list(set(ages_))))  # Gyr
    ages = ages_Gyr * 1e9 * yr

    lam = lam_[ages_ == ages_[0]] * Angstrom

    spec = np.zeros((len(ages), len(metallicities), len(lam)))

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename)

    # at each point in spec convert the units
    for iZ, Z in enumerate(metallicities):
        for ia, age_Gyr in enumerate(ages_Gyr):
            fn = (
                f"{input_dir}/sed_M13.{imf_code[imf]}"
                f"z{metallicity_codes[Z]}"
            )
            print(iZ, ia, fn)
            ages_, _, lam_, llam_ = np.loadtxt(fn).T

            llam = llam_[ages_ == age_Gyr] * erg / s / Angstrom
            lnu = llam_to_lnu(lam, llam)
            spec[ia, iZ] = lnu

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
        log_on_read=log_on_read,
        alt_axes=("log10ages", "metallicities"),
        weight="initial_masses",
    )

    # Include the specific ionising photon luminosity
    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="Maraston+13 download and grid creation")

    args = parser.parse_args()

    grid_dir = args.grid_dir

    # Define the model metadata
    sps_name = "maraston13"
    imfs = ["salpeter", "kroupa"]
    imf_code = {"salpeter": "ss", "kroupa": "kr"}

    input_dir = args.input_dir
    input_dir += f"/{sps_name}"

    # create directory to store downloaded output if it doesn't exist
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    # then run the rest
    for imf in imfs:
        
        model = {
            "sps_name": sps_name,
            "sps_version": False,
            "sps_variant": False,
            "imf_type": imf,
            "imf_masses": [0.1, 100],
            "imf_slopes": False,
            "alpha": False,
            }
        
        synthesizer_model_name = get_model_filename(model)
        print(synthesizer_model_name)
                
        make_grid(synthesizer_model_name, imf, input_dir, grid_dir)
