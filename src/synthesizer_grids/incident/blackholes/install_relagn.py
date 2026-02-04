"""
Create a synthesizer incident grid for the RELAGN model of Hagen and Done
(2023) (https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.3455H/abstract).

This uses the relagn module (https://github.com/scotthgn/RELAGN.git). To use
this we first need to download it by cloning the github
repo, i.e.:
git clone https://github.com/scotthgn/RELAGN.git
Since it doesn't have an __init__.py we need to add the path to our python
path. Please follow instructions in the relagn README for this.
This also requires that xspec (https://heasarc.gsfc.nasa.gov/xanadu/xspec/) is
installed.
"""

import itertools
from multiprocessing import Pool

import numpy as np
import yaml

# Import relagn module
from relagn import relagn
from scipy.optimize import brentq
from unyt import Angstrom, Hz, Msun, dimensionless, erg, s

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser


def calc_risco(spin):
    """Calculating innermost stable circular orbit for a spinning
    black hole.

    Attrs:
        spin (float): Dimensionless spin parameter (-1 to 1)

    Returns:
        risco (float): Innermost stable circular orbit in units of GM/c^2
    """
    Z1 = 1 + (1 - spin**2) ** (1 / 3) * (
        (1 + spin) ** (1 / 3) + (1 - spin) ** (1 / 3)
    )
    Z2 = np.sqrt(3 * spin**2 + Z1**2)

    risco = 3 + Z2 - np.sign(spin) * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))

    return risco


def calc_efficiency(spin):
    """Calculate the accretion efficiency for a given spin.

    Attrs:
        spin (float): Dimensionless spin parameter (-1 to 1)

    Returns:
        eta (float): Accretion efficiency
    """
    risco = calc_risco(spin)

    eta = 1 - np.sqrt(1 - 2 / (3 * risco))

    return eta


def invert_spin(eta_target, spin_min=-1.0, spin_max=1.0):
    """Invert efficiency to get spin.

    Attrs:
        eta_target (float): Accretion efficiency (0 to 1)
        spin_min (float): Minimum spin to search
        spin_max (float): Maximum spin to search

    Returns:
        spin (float): Dimensionless spin parameter (-1 to 1)
    """

    def f(spin):
        return eta_target - calc_efficiency(spin)

    return brentq(f, spin_min, spin_max, xtol=1e-12, rtol=1e-12)


def compute_single_spectrum(mass, mdot, spin, cos_inc):
    """Function to calculate SED for a single set of parameters.

    Attrs:
        mass (float): Black hole mass in solar masses.
        mdot (float): Accretion rate in Eddington units.
        spin (float): Dimensionless spin parameter.
        cos_inc (float): Cosine of the inclination angle.

    Returns:
        lnu (np.ndarray): The computed SED in wavelength order.
    """
    # Re-instantiate the physics object inside the worker process
    dagn = relagn(
        a=spin,
        cos_inc=cos_inc,
        log_mdot=np.log10(mdot),
        M=mass,
    )

    # Get relativistic sed
    lnu = dagn.get_totSED(rel=True)

    # Return reversed array (wavelength order)
    return lnu[::-1]


if __name__ == "__main__":
    """
    Create incident AGN spectra assuming the RELAGN model.
    """

    axes_names = [
        "masses",
        "accretion_rates_eddington",
        "spins",
        "cosine_inclinations",
    ]

    axes_descriptions = {
        "masses": "blackhole mass",
        "accretion_rates_eddington": "BH accretion rate / Eddington accretion"
        " rate [LEdd=eta MdotEdd c^2]",
        "cosine_inclinations": "cosine of the inclination",
        "spins": "dimensionless spin",
    }

    axes_units = {
        "masses": Msun,
        "accretion_rates_eddington": dimensionless,
        "cosine_inclinations": dimensionless,
        "spins": dimensionless,
    }

    # Set log_on_read, i.e. which axes should be logged when extracted
    # cosine_inclination is set later
    log_on_read = {
        "masses": True,
        "accretion_rates_eddington": True,
        "spins": False,
    }

    # Parameter ranges
    cosine_inclination_range = [0.09, 1.0]
    spin_range = [0.0, 0.998]

    # Set up the command line arguments
    parser = Parser(description="RELAGN AGN model creation.")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    # number of processes to use
    parser.add_argument("--num-procs", type=int, default=8)

    # get the arguments
    args = parser.parse_args()

    # open the config file and extract parameters
    with open(args.config_file, "r") as file:
        parameters = yaml.safe_load(file)

    model_name = parameters["model"]
    mass = 10 ** np.array(parameters["log10masses"])
    accretion_rate_eddington = 10 ** np.array(
        parameters["log10accretion_rates_eddington"]
    )
    if "spins" not in parameters:
        rad_efficiencies = np.array(parameters["radiative_efficiencies"])
        spin = np.array([invert_spin(eta) for eta in rad_efficiencies])
        print(f"spin={spin} for rad_efficiencies={rad_efficiencies}")
    else:
        spin = np.array(parameters["spins"])

    # check whether isotropic or not
    if isinstance(parameters["cosine_inclinations"], float):
        cosine_inclination = parameters["cosine_inclinations"]
        axes_names.remove("cosine_inclinations")
        isotropic = True
    else:
        cosine_inclination = np.array(parameters["cosine_inclinations"])
        isotropic = False

    # Check if parameters within range
    if np.any(spin < spin_range[0]) or np.any(spin > spin_range[1]):
        raise ValueError(
            f"Spin values must be within range {spin_range}, "
            f"but got spins={spin}"
        )

    if np.any(cosine_inclination < cosine_inclination_range[0]) or np.any(
        cosine_inclination > cosine_inclination_range[1]
    ):
        raise ValueError(
            f"Cosine inclination values must be within range "
            f"{cosine_inclination_range}, but got "
            f"cosine_inclinations="
            f"{cosine_inclination}"
        )

    # Model defintion dictionary
    model = {
        "name": model_name,
        "type": "agn",
        "family": "relagn",
    }

    # Define the grid filename and path
    out_filename = f"{args.grid_dir}/{model_name}.hdf5"

    axes_values = {
        "masses": mass,
        "accretion_rates_eddington": accretion_rate_eddington,
        "spins": spin,
    }

    if not isotropic:
        axes_values["cosine_inclinations"] = np.array(cosine_inclination)

    # the shape of the grid (useful for creating outputs)
    axes_shape = list(
        [len(axes_values[axis_name]) for axis_name in axes_names]
    )

    # define axes dictionary which is saved to the HDF5 file
    axes = {}
    for axis_name in axes_names:
        axes[axis_name] = axes_values[axis_name] * axes_units[axis_name]
        axes[axis_name] = axes[axis_name].to(axes_units[axis_name])

    print(axes_values)
    print(axes)

    # initialise default model, to get wavelength grid
    dagn = relagn()
    lam = dagn.wave_grid[::-1] * Angstrom

    iterables = [values for values in axes_values.values()]
    grid_shape = tuple(len(x) for x in iterables if len(x) > 1)

    # Generate all combinations of arguments (Flattening the loops)
    # This creates a generator, which is memory efficient
    param_grid = itertools.product(*iterables)

    print("Starting RELAGN SED calculations...")
    with Pool(processes=args.num_procs) as pool:
        # starmap unpacks the tuple arguments for us: func(*args)
        results = pool.starmap(compute_single_spectrum, param_grid)

    # Reshape results into the final grid shape
    final_shape = grid_shape + (len(lam),)
    spec = np.array(results).reshape(final_shape)

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename)

    if not isotropic:
        log_on_read["cosine_inclinations"] = False

    if "spins" not in parameters:
        axes_names.remove("spins")
        del axes["spins"]
        del log_on_read["spins"]

        if (
            np.isscalar(rad_efficiencies) is False
            and np.size(rad_efficiencies) != 1
        ):
            axes["radiative_efficiencies"] = rad_efficiencies * dimensionless
            axes_descriptions["radiative_efficiencies"] = (
                "radiative efficiency of the accreting black hole"
            )
            log_on_read["radiative_efficiencies"] = False

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
