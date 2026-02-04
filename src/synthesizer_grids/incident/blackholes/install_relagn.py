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
from numpy.typing import NDArray

# Import relagn module
from relagn import relagn
from scipy.optimize import brentq
from synthesizer.exceptions import InconsistentParameter
from unyt import Angstrom, Hz, Msun, dimensionless, erg, s

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser


def rename_dictionary_key(
    dictionary: dict,
    old_key: str,
    new_key: str,
    new_value=None,
) -> dict:
    """Rename a key in a dictionary.

    Attrs:
        dictionary (dict): Input dictionary
        old_key (str): Old key name
        new_key (str): New key name
        new_value: New value for the renamed key

    Returns:
        dictionary (dict): Updated dictionary
    """

    return {
        (new_key if k == old_key else k): (new_value if k == old_key else v)
        for k, v in dictionary.items()
    }


def calc_risco(spin: float) -> float:
    """Calculating innermost stable circular orbit for a spinning
    black hole.

    Attrs:
        spin (float): Dimensionless spin parameter (-0.998 to 0.998)

    Returns:
        risco (float): Innermost stable circular orbit in units of GM/c^2
    """
    Z1 = 1 + (1 - spin**2) ** (1 / 3) * (
        (1 + spin) ** (1 / 3) + (1 - spin) ** (1 / 3)
    )
    Z2 = np.sqrt(3 * spin**2 + Z1**2)

    risco = 3 + Z2 - np.sign(spin) * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))

    return risco


def calc_efficiency(spin: float) -> float:
    """Calculate the accretion efficiency for a given spin.

    Attrs:
        spin (float): Dimensionless spin parameter (0.0 to 0.998)

    Returns:
        eta (float): Accretion efficiency
    """
    risco = calc_risco(spin)

    eta = 1 - np.sqrt(1 - 2 / (3 * risco))

    return eta


def invert_spin(
    eta_target: float, spin_min: float = 0.0, spin_max: float = 0.998
) -> float:
    """Invert efficiency to get spin.

    Attrs:
        eta_target (float): Accretion efficiency (0 to 1)
        spin_min (float): Minimum spin to search, RELAGN only
        valid for prograde spins, so min is 0.0
        spin_max (float): Maximum spin to search

    Returns:
        spin (float): Dimensionless spin parameter (0.0 to 0.998)

    Raises:
        ValueError: If eta_target is outside achievable spin range.
    """

    eta_min = calc_efficiency(spin_min)
    eta_max = calc_efficiency(spin_max)
    if not (eta_min <= eta_target <= eta_max):
        raise InconsistentParameter(
            f"eta_target={eta_target} outside achievable range "
            f"[{eta_min:.4f}, {eta_max:.4f}] for spin in "
            f"[{spin_min}, {spin_max}]"
        )

    def func(spin: float) -> float:
        """Function to find root of."""
        return eta_target - calc_efficiency(spin)

    return brentq(func, spin_min, spin_max, xtol=1e-12, rtol=1e-12)


def compute_single_spectrum(
    mass: float,
    mdot: float,
    spin: float,
    cos_inc: float,
) -> NDArray:
    """Function to calculate SED for a single set of parameters.

    Attrs:
        mass (float): Black hole mass in solar masses.
        mdot (float): Accretion rate in Eddington units.
        spin (float): Dimensionless spin parameter.
        cos_inc (float): Cosine of the inclination angle.

    Returns:
        lnu (NDArray): The computed SED in wavelength order.
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
        "radiative_efficiencies": "radiative efficiency of the accreting "
        "black hole",
    }

    axes_units = {
        "masses": Msun,
        "accretion_rates_eddington": dimensionless,
        "cosine_inclinations": dimensionless,
        "spins": dimensionless,
        "radiative_efficiencies": dimensionless,
    }

    # Set log_on_read, i.e. which axes should be logged when extracted
    # cosine_inclination is set later
    log_on_read = {
        "masses": True,
        "accretion_rates_eddington": True,
        "spins": False,
        "radiative_efficiencies": False,
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
    cos_inc_param = parameters["cosine_inclinations"]
    if np.isscalar(cos_inc_param):
        cosine_inclination = float(cos_inc_param)
        axes_names.remove("cosine_inclinations")
        isotropic = True
    else:
        cosine_inclination = np.array(cos_inc_param)
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
    if isotropic:
        # When isotropic, add the scalar cosine_inclination to
        # each parameter tuple
        param_grid = [
            (*params, cosine_inclination)
            for params in itertools.product(*iterables)
        ]
    else:
        param_grid = list(itertools.product(*iterables))
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
        if np.atleast_1d(rad_efficiencies).size > 1:
            # Add radiative efficiencies to axes and descriptions.
            # We need to rename the spins axis to radiative_efficiencies,
            # and ensure the same orders of values are maintained since
            # the order was used to set the spectra shape
            axes = rename_dictionary_key(
                axes,
                old_key="spins",
                new_key="radiative_efficiencies",
                new_value=rad_efficiencies * dimensionless,
            )

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
