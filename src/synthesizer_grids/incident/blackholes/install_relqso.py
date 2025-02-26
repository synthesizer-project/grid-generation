"""
Create a synthesizer incident grid for the RELQSO model of Hagen and Done
(2023) (https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.3455H/abstract).

This uses the relagn module (https://github.com/scotthgn/RELAGN.git). To use
this we first need to download it by cloning the github
repo, i.e.:
git clone https://github.com/scotthgn/RELAGN.git
Since it doesn't have an __init__.py we need to add the path to our python
path.
This also requires that xspec (https://heasarc.gsfc.nasa.gov/xanadu/xspec/) is
installed.
"""

import sys

import numpy as np
import yaml
from unyt import Angstrom, Hz, Msun, erg, s

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser

sys.path.append("RELAGN/src/python_version")

if __name__ == "__main__":
    # Import relagn module
    from relagn import relagn  # noqa: E402

    """
    Create incident AGN spectra assuming the RELQSO model.
    """

    axes_names = [
        "mass",
        "accretion_rate_eddington",
        "spin",
        "cosine_inclination",
    ]

    axes_descriptions = {
        "mass": "blackhole mass",
        "accretion_rate_eddington": "BH accretion rate / Eddington accretion"
        " rate [LEdd=eta MdotEdd c^2]",
        "cosine_inclination": "cosine of the inclination",
        "spin": "dimensionless spin",
    }

    axes_units = {
        "mass": Msun,
        "accretion_rate_eddington": None,
        "cosine_inclination": None,
        "spin": None,
    }

    # Set up the command line arguments
    parser = Parser(description="RELQSO AGN model creation.")

    # parameter file to use
    parser.add_argument("-config_file", type=str, required=True)

    # get the arguments
    args = parser.parse_args()

    # open the config file and extract parameters
    with open(args.config_file, "r") as file:
        parameters = yaml.safe_load(file)

    model_name = parameters["model"]
    mass = 10 ** np.array(parameters["log10_mass"])
    accretion_rate_eddington = 10 ** np.array(
        parameters["log10_accretion_rate_eddington"]
    )
    spin = np.array(parameters["spin"])

    # check whether isotropic or not
    if isinstance(parameters["cosine_inclination"], float):
        cosine_inclination = parameters["cosine_inclination"]
        axes_names.remove("cosine_inclination")
        isotropic = True
    else:
        cosine_inclination = np.array(parameters["cosine_inclination"])
        isotropic = False

    # Model defintion dictionary
    model = {
        "name": model_name,
        "type": "agn",
        "family": "relqso",
    }

    # Define the grid filename and path
    out_filename = f"{args.grid_dir}/{model_name}.hdf5"

    axes_values = {
        "mass": mass,
        "accretion_rate_eddington": accretion_rate_eddington,
        "spin": spin,
    }

    if not isotropic:
        axes_values["cosine_inclination"] = np.array(cosine_inclination)

    # the shape of the grid (useful for creating outputs)
    axes_shape = list(
        [len(axes_values[axis_name]) for axis_name in axes_names]
    )

    # define axes dictionary which is saved to the HDF5 file
    axes = {}
    for axis_name in axes_names:
        axis_values = axes_values[axis_name]
        if axes_units[axis_name] is not None:
            axes[axis_name] = axis_values  # * axes_units[axis_name]
        # assumed to be dimensionless
        else:
            axes[axis_name] = axis_values

    print(axes_shape)
    print(axes_values)

    # initialise default model, to get wavelength grid
    dagn = relagn()
    lam = dagn.wave_grid[::-1] * Angstrom

    # create empty spectra grid
    spec = np.zeros((*axes_shape, len(lam)))

    for i1, mass_ in enumerate(axes_values["mass"]):
        for i2, accretion_rate_eddington_ in enumerate(
            axes_values["accretion_rate_eddington"]
        ):
            for i3, spin_ in enumerate(axes_values["spin"]):
                if isotropic:
                    dagn = relagn(
                        a=spin_,
                        cos_inc=cosine_inclination,
                        log_mdot=np.log10(accretion_rate_eddington_),
                        M=mass_,
                    )

                    # get relativistic sed
                    lnu = dagn.get_totSED(rel=True)

                    # record spectra, making sure to reverse it so it is
                    # wavelength order.
                    spec[i1, i2, i3] = lnu[::-1]

                else:
                    for i4, cosine_inclination_ in enumerate(
                        axes_values["cosine_inclination"]
                    ):
                        dagn = relagn(
                            a=spin_,
                            cos_inc=cosine_inclination_,
                            log_mdot=np.log10(accretion_rate_eddington_),
                            M=mass_,
                        )

                        # get relativistic sed
                        lnu = dagn.get_totSED(rel=True)

                        # record spectra, making sure to reverse it so it is
                        # wavelength order.
                        spec[i1, i2, i3, i4] = lnu[::-1]

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename, mode="w", overwrite=True)

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model,
        axes=axes,
        descriptions=axes_descriptions,
        wavelength=lam,
        spectra={"incident": spec * erg / s / Hz},
        weight="bolometric_luminosities",
    )

    # Include the specific ionising photon luminosity
    print("Calculating and saving specific ionising luminosity")
    out_grid.add_specific_ionising_lum()
