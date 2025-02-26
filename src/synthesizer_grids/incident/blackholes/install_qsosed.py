"""
Create a synthesizer incident grid for the QSOSED model of Kubota and Done
(2018) (https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.1247K/abstract).

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
from unyt import Angstrom, Hz, Msun, c, dimensionless, erg, s

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser

sys.path.append("RELAGN/src/python_version")

if __name__ == "__main__":
    # Import relagn module
    from relagn import relagn  # noqa: E402

    """
    Create incident AGN spectra assuming the QSOSED model.
    """

    axes_names = ["mass", "accretion_rate_eddington", "cosine_inclination"]

    axes_descriptions = {
        "mass": "blackhole mass",
        "accretion_rate_eddington": "BH accretion rate / Eddington accretion"
        " rate [LEdd=eta MdotEdd c^2]",
        "cosine_inclination": "cosine of the inclination",
    }

    axes_units = {
        "mass": Msun,
        "accretion_rate_eddington": dimensionless,
        "cosine_inclination": dimensionless,
    }

    # Set up the command line arguments
    parser = Parser(description="QSOSED AGN model creation.")

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
        "family": "qsosed",
    }

    # Define the grid filename and path
    out_filename = f"{args.grid_dir}/{model_name}.hdf5"

    axes_values = {
        "mass": mass,
        "accretion_rate_eddington": accretion_rate_eddington,
    }

    if not isotropic:
        axes_values["cosine_inclination"] = cosine_inclination

    # the shape of the grid (useful for creating outputs)
    axes_shape = list(
        [len(axes_values[axis_name]) for axis_name in axes_names]
    )

    # define axes dictionary which is saved to the HDF5 file
    axes = {}
    for axis_name in axes_names:
        axis_values = axes_values[axis_name]
        if axes_units[axis_name] is not None:
            axes[axis_name] = axis_values * axes_units[axis_name]
        # assumed to be dimensionless
        else:
            axes[axis_name] = axis_values

    # initialise default model, to get wavelength grid
    dagn = relagn()
    lam = dagn.wave_grid[::-1] * Angstrom
    nu = c / lam
    nu_hz = nu.to("Hz").value

    # create empty spectra grid
    spec = np.zeros((*axes_shape, len(lam)))

    for i1, mass_ in enumerate(axes_values["mass"]):
        for i2, accretion_rate_eddington_ in enumerate(
            axes_values["accretion_rate_eddington"]
        ):
            if isotropic:
                # spin is assumed to be zero here
                dagn = relagn(
                    a=0.0,
                    cos_inc=cosine_inclination,
                    log_mdot=np.log10(accretion_rate_eddington_),
                    M=mass_,
                )

                # lnu = dagn.get_totSED(rel=True) # relativistic
                lnu = dagn.get_totSED(rel=False)  # non-relativistic

                spec[i1, i2] = lnu[::-1]

            else:
                for i3, cosine_inclination_ in enumerate(
                    axes_values["cosine_inclination"]
                ):
                    # spin is assumed to be zero here
                    dagn = relagn(
                        a=0.0,
                        cos_inc=cosine_inclination_,
                        log_mdot=np.log10(accretion_rate_eddington_),
                        M=mass_,
                    )

                    # lnu = dagn.get_totSED(rel=True) # relativistic
                    lnu = dagn.get_totSED(rel=False)  # non-relativistic

                    spec[i1, i2, i3] = lnu[::-1]

    # Normalise the spectra using the isotropic spectra (i.e.
    # cosine_inclination=0.5)

    # If only a grid containing isoptropic spectra this simply means
    # normalising every spectra to unit
    if isotropic:
        # loop over each axis
        for i1, mass_ in enumerate(axes_values["mass"]):
            for i2, accretion_rate_eddington_ in enumerate(
                axes_values["accretion_rate_eddington"]
            ):
                # determine the boloe
                bolometric_luminosity = -np.trapezoid(spec[i1, i2], nu_hz)
                spec[i1, i2] /= bolometric_luminosity

    # Otherwise identify the index correspinding and divide all by this
    else:
        # Find the index corresponding to cosine_inclination=0.5
        isotropic_index = np.where(cosine_inclination == 0.5)[0]

        # Loop over each axis
        for i1, mass_ in enumerate(axes_values["mass"]):
            for i2, accretion_rate_eddington_ in enumerate(
                axes_values["accretion_rate_eddington"]
            ):
                # Determine the bolometric luminosity
                bolometric_luminosity = -np.trapz(
                    spec[i1, i2, isotropic_index], nu_hz
                )

                for i3, cosine_inclination_ in enumerate(
                    axes_values["cosine_inclination"]
                ):
                    spec[i1, i2, i3] /= bolometric_luminosity

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename)

    log_on_read = {"mass": True, "accretion_rate_eddington": True}

    if not isotropic:
        log_on_read["cosine_inclination"] = False

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model,
        axes=axes,
        descriptions=axes_descriptions,
        wavelength=lam,
        log_on_read=log_on_read,
        spectra={"incident": spec * erg / s / Hz},
        weight="bolometric_luminosities",
    )

    # Include the specific ionising photon luminosity
    print("Calculating and saving specific ionising luminosity")
    out_grid.add_specific_ionising_lum()
