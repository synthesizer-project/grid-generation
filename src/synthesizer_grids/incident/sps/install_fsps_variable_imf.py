"""
Create FSPS synthesizer grids including a variable high-mass slope as an
additional axis.

Example:
    python install_fsps_variable_imf.py \
    --input-dir path/to/input/dir \
    --grid-dir path/to/grid/dir \
"""

import fsps
import numpy as np
from unyt import Hz, angstrom, dimensionless, erg, s, yr
from utils import get_model_filename

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser


def generate_grid(model):
    """Main function to create fsps grids used by synthesizer"""

    # Extra list of high mass slopes
    high_mass_slopes = model["high_mass_slope"]

    # Generate the synthesizer_model_name
    synthesizer_model_name = get_model_filename(model)

    # This is the full path to the ultimate HDF5 grid file
    out_filename = f"{grid_dir}/{synthesizer_model_name}.hdf5"

    # Let's initialise one model to extract some of the information we need
    sp = fsps.StellarPopulation(
        imf_type=2,
        imf_upper_limit=model["imf_masses"][-1],
        imf_lower_limit=model["imf_masses"][0],
        imf1=model["imf_slopes"][0],
        imf2=2.3,
        imf3=2.3,)

    # Update the sps_variant parameter with the actual model used
    model["sps_variant"] = "-".join(
        map(lambda x: x.decode("utf-8"), sp.libraries[:2])
    )
    print(model["sps_variant"])

    lam = sp.wavelengths  # units: Angstroms
    log10ages = sp.log_age  # units: log10(years)
    ages = 10**log10ages
    metallicities = sp.zlegend  # units: log10(metal)

    na = len(log10ages)
    nmetal = len(metallicities)

    # Define the output array
    spec = np.zeros((na, nmetal, len(high_mass_slopes), len(lam)))

    for high_mass_slope_index, high_mass_slope in enumerate(high_mass_slopes):

        print(high_mass_slope)

        # NOTE this imf_type (2) assumes the IMF has three slopes with the
        # boundaries (mlow, 0.5, 1.0, mup). In this model we are only
        # interested in varying high-mass slope above 0.5 and so we can
        # safely set imf2 and imf3 to our chosen high-mass slope.

        # Initialise the FSPS model for the particular choice of high mass
        # slope
        sp = fsps.StellarPopulation(
            imf_type=2,
            imf_upper_limit=model["imf_masses"][-1],
            imf_lower_limit=model["imf_masses"][0],
            imf1=model["imf_slopes"][0],
            imf2=high_mass_slope,
            imf3=high_mass_slope,)

        # Loop over metallicity
        for metallicity_index in range(nmetal):
            spec_ = sp.get_spectrum(zmet=metallicity_index + 1)[1]

            # Loop over age
            for age_index in range(na):

                # Extract spectra in Lsol / Hz
                lnu = spec_[age_index]

                # Convert to erg s^-1 Hz^-1 Msol^-1
                lnu *= 3.826e33

                # Assign to array
                spec[age_index, metallicity_index, high_mass_slope_index] = lnu

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename)

    # A dictionary with Boolean values for each axis, where True
    # indicates that the attribute should be interpolated in
    # logarithmic space.
    log_on_read = {
        "ages": True,
        "metallicities": False,
        "high_mass_slope": False}

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
            "high_mass_slope": high_mass_slopes * dimensionless,
        },
        wavelength=lam * angstrom,
        spectra={"incident": spec * erg / s / Hz},
        alt_axes=("log10ages", "metallicities", "high_mass_slopes"),
        descriptions={
            "high_mass_slope": "high mass (>0.5 Msun) slope of the IMF"},
        log_on_read=log_on_read,
    )

    # Include the specific ionising photon luminosity
    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="FSPS download and grid creation")

    # Unpack the arguments
    args = parser.parse_args()
    grid_dir = args.grid_dir

    # High-mass slopes
    high_mass_slopes = np.arange(1.3, 3.01, 0.1)
    # high_mass_slopes = np.array([1.3, 2.3])

    # No download for FSPS
    if args.download:
        print("FSPS is a python package, no download required")

    # Define the parameters of the defaul model. Variants simply update these
    # parameters as required.
    model = {
        "sps_name": "fsps",
        # this is set later depending on the isochrones/spectra used
        "sps_variant": False,
        "imf_type": "bpl",  # named IMF or bpl (broken power law)
        "imf_masses": [0.08, 0.5, 100],
        "imf_slopes": [1.3, ],
        "alpha": False,
        "pyfsps_version": str(fsps.__version__),
        "sps_version": "3.2",
        "high_mass_slope": high_mass_slopes,
    }

    generate_grid(model)
