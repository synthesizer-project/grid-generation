"""
Download the Maraston2024 SPS model
and convert to HDF5 synthesizer grid.
"""

import os

import numpy as np
from synthesizer.conversions import llam_to_lnu
from unyt import Angstrom, Hz, dimensionless, erg, s, yr
from utils import get_model_filename

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser


def make_grid(
    synthesizer_model_name, rotation, model_type, imf, input_dir, grid_dir
):
    """Main function to convert Maraston 2024 and
    produce grids used by synthesizer
    Args:
        synthesizer_model_name (string):
            name of the model to be saved.
        rotation (string):
            value of stellar rotation for the model,
            "0.00" for no rotation or "0.40" for rotation.
        model_type (string):
            either "" for standard model or "Tenc" for non-corrected
            effective temperatures.
        imf (string):
            either "kr" for a Kroupa IMF or "ss" for Salpeter.
        input_dir (string):
            directory where the raw Maraston+24 files are read from.
        grid_dir (string):
            directory where the grids are created.
    Returns:
        fname (string):
            output filename
    """

    # Array of available metallicities
    metallicities = np.array([0.0003, 0.002, 0.006, 0.014, 0.02])

    # Codes for converting metallicty
    metallicity_code = {
        0.0003: "-1.7",
        0.002: "-1.35",
        0.006: "-0.33",
        0.014: "+0.00",  # solar metallicity
        0.02: "+0.35",
    }

    if model_type == "Tenc":
        model_type = "_Tenc"

    # Open first raw data file to get age
    fn = (
        f"{input_dir}/sed_ssp_M24_vini{rotation}{model_type}_{imf}"
        f"{metallicity_code[metallicities[0]]}"
    )

    ages_, _, lam_, llam_ = np.loadtxt(fn).T  # llam is in (ergs /s /AA /Msun)

    ages_Gyr = np.sort(np.array(list(set(ages_))))  # Gyr

    # Convert units to years
    ages = ages_Gyr * 1e9

    # Get wavelengths for first age
    lam = lam_[ages_ == ages_[0]] * Angstrom

    spec = np.zeros((len(ages), len(metallicities), len(lam)))

    # Create the GridFile ready to take outputs
    out_grid = GridFile(f"{grid_dir}/{synthesizer_model_name}.hdf5")

    log_on_read = {"ages": True, "metallicities": False}

    # At each point in spec convert the units
    for imetal, metallicity in enumerate(metallicities):
        for ia, age_Gyr in enumerate(ages_Gyr):
            fn = (
                f"{input_dir}/sed_ssp_M24_vini{rotation}{model_type}_{imf}"
                f"{metallicity_code[metallicity]}"
            )
            ages_, _, lam_, llam_ = np.loadtxt(fn).T

            llam = llam_[ages_ == age_Gyr] * erg / s / Angstrom

            lnu = llam_to_lnu(lam, llam)
            spec[ia, imetal] = lnu

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=lam,
        spectra={"incident": spec * erg / s / Hz},  # check this unit
        alt_axes=("log10ages", "metallicities"),
        log_on_read=log_on_read,
        weight="initial_masses",
    )

    # Include the specific ionising photon luminosity
    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="Maraston+24 download and grid creation")

    args = parser.parse_args()

    grid_dir = args.grid_dir

    # Define the model metadata
    sps_name = "maraston24"
    rotations = ["0.00", "0.40"]
    model_types = ["", "Tenc"]
    imfs = ["kr", "ss"]

    input_dir = f"{args.input_dir}/{sps_name}"

    # Create directory to store downloaded output if it doesn't exist
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    for imf in imfs:
        for model_type in model_types:
            for rotation in rotations:
                if imf == "kr":
                    imf_type = "kroupa"
                if imf == "ss":
                    imf_type = "salpeter"

                if model_type == "Tenc":
                    variant_name = f"Tenc_{rotation}"
                if model_type == "":
                    variant_name = rotation

                model = {
                    "sps_name": sps_name,
                    "sps_version": False,
                    "sps_variant": variant_name,
                    "imf_type": imf_type,
                    "imf_masses": [0.1, 100],
                    "imf_slopes": False,
                    "alpha": False,
                }

                synthesizer_model_name = get_model_filename(model)
                print(synthesizer_model_name)

                make_grid(
                    synthesizer_model_name,
                    rotation,
                    model_type,
                    imf,
                    input_dir,
                    grid_dir,
                )
