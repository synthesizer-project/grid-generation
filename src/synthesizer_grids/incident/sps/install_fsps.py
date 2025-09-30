"""
Create FSPS synthesizer grids.

Example:
    python install_fsps.py \
    --input-dir path/to/input/dir \
    --grid-dir path/to/grid/dir \
    --include-chabrier \
    --include-imf-variants \
"""

import fsps
import numpy as np
from unyt import Hz, angstrom, dimensionless, erg, s, yr
from utils import get_model_filename

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser


def generate_grid(model):
    """Main function to create fsps grids used by synthesizer"""

    # set up StellarPopulation grid
    imf_type = model["imf_type"]
    imf_masses = model["imf_masses"]

    if imf_type == "chabrier03":
        sp = fsps.StellarPopulation(
            imf_type=1,
            imf_upper_limit=imf_masses[-1],
            imf_lower_limit=imf_masses[0],
        )
    elif imf_type == "bpl":
        imf_slopes = model["imf_slopes"]
        # NOTE THIS ASSUMES boundaries of 0.5, 1.0

        if (imf_masses[1] != 0.5) or (imf_masses[2] != 1.0):
            # raise exception
            print(
                """WARNING: this IMF definition requires that the boundaries
                are [m_low, 0.5, 1.0, m_high]"""
            )

        sp = fsps.StellarPopulation(
            imf_type=2,
            imf_upper_limit=imf_masses[-1],
            imf_lower_limit=imf_masses[0],
            imf1=imf_slopes[0],
            imf2=imf_slopes[1],
            imf3=imf_slopes[2],
        )
    else:
        # raise exception
        pass

    # include the isochrones/spectra as the sps_variant name
    # model['sps_variant'] = '-'.join(map(lambda x: str(x), sp.libraries[:2]))

    model["sps_variant"] = "-".join(
        map(lambda x: x.decode("utf-8"), sp.libraries[:2])
    )
    print(model["sps_variant"])

    # generate the synthesizer_model_name
    synthesizer_model_name = get_model_filename(model)

    # this is the full path to the ultimate HDF5 grid file
    out_filename = f"{grid_dir}/{synthesizer_model_name}.hdf5"

    lam = sp.wavelengths  # units: Angstroms
    log10ages = sp.log_age  # units: log10(years)
    ages = 10**log10ages
    metallicities = sp.zlegend  # units: log10(metal)

    na = len(log10ages)
    nmetal = len(metallicities)

    spec = np.zeros((na, nmetal, len(lam)))
    stellar_fraction = np.zeros((na, nmetal))

    for imetal in range(nmetal):
        spec_ = sp.get_spectrum(zmet=imetal + 1)[1]  # 2D array Lsol / AA
        _mfrac = sp.stellar_mass / sp.formed_mass
        for ia in range(na):
            lnu = spec_[ia]  # Lsol / Hz
            lnu *= 3.826e33  # erg s^-1 Hz^-1 Msol^-1
            spec[ia, imetal] = lnu
            stellar_fraction[ia, imetal] = _mfrac[ia]

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename)

    print("metallicities:", metallicities)

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
        wavelength=lam * angstrom,
        spectra={"incident": spec * erg / s / Hz},
        alt_axes=("log10ages", "metallicities"),
        log_on_read=log_on_read,
        weight="initial_masses",
    )

    # Write datasets specific to fsps
    out_grid.write_dataset(
        "star_fraction",
        stellar_fraction * dimensionless,
        "Two-dimensional remaining stellar fraction grid, [age, Z]",
        log_on_read=False,
    )

    # Include the specific ionising photon luminosity
    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="FSPS download and grid creation")

    # Include Chabrier IMF
    parser.add_argument(
        "--include-chabrier",
        action="store_true",
        help="Include Chabrier?",
    )

    # Include IMF variants
    parser.add_argument(
        "--include-imf-variants",
        action="store_true",
        help="Include IMF variants?",
    )

    # Unpack the arguments
    args = parser.parse_args()
    grid_dir = args.grid_dir
    include_chabrier = args.include_chabrier
    include_imf_variants = args.include_imf_variants

    # No download for FSPS
    if args.download:
        print("FSPS is a python package, no download required")

    # Define the parameters of the defaul model. Variants simply update these
    # parameters as required.
    default_model = {
        "sps_name": "fsps",
        # this is set later depending on the isochrones/spectra used
        "sps_variant": False,
        "imf_type": "bpl",  # named IMF or bpl (broken power law)
        "imf_masses": [0.08, 0.5, 1, 120],
        "imf_slopes": [1.3, 2.3, 2.3],
        "alpha": False,
        "pyfsps_version": str(fsps.__version__),
        "sps_version": "3.2",
    }

    # List of models to create
    models = []

    # Add default model
    models += [{}]

    # Add Chabrier IMF model
    if include_chabrier:
        models += [
            {
                "imf_type": "chabrier03",
                "imf_masses": [0.08, 120],
                "imf_slopes": [],
            },
        ]

    # Include IMF variants
    if include_imf_variants:
        # Add models with varying high-mass slopes
        models += [
            {"imf_slopes": [1.3, 2.3, a3]} for a3 in np.arange(1.5, 3.01, 0.1)
        ]

        # Add Chabrier models with varying high-mass cut-offs
        models += [
            {"imf_type": "chabrier03", "imf_masses": [0.08, hmc]}
            for hmc in [1, 2, 5, 10, 20, 50, 100]
        ]

        # Add Chabrier models with varying high-mass cut-offs
        models += [
            {"imf_type": "chabrier03", "imf_masses": [lmc, 120]}
            for lmc in [0.5, 1, 2, 5, 10, 20, 50]
        ]

    # Loop over models
    for model_ in models:
        # Combine default model parameters with variants
        model = default_model | model_
        print(model)

        # Make grid
        generate_grid(model)
