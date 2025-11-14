"""
Create Cloudy input scripts by Sobol-sampling both incident grid parameters
(age, metallicity) and photoionization parameters.

Uses a single Sobol sequence across all parameters, then interpolates spectra
from the incident grid at the sampled age/metallicity points.
"""

import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.stats import qmc
from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.particle import Stars
from synthesizer.photoionisation import cloudy17, cloudy23
from unyt import Msun, yr

import synthesizer_grids.cloudy.submission_scripts as submission_scripts
from synthesizer_grids.parser import Parser


def parse_sobol_params(params, incident_grid):
    """
    Parse parameters into incident, photoionization, and fixed categories.
    Automatically uses incident grid axes ranges when not specified in params.

    Args:
        params (dict): Parameter dictionary from YAML file
        incident_grid (Grid): Incident grid to extract axes ranges

    Returns:
        tuple: (incident_params, photoionisation_params, fixed_params)
            - incident_params: dict mapping axes to (min, max, scale)
            - photoionisation_params: dict mapping params to (min, max, scale)
            - fixed_params: dict of fixed parameter values
    """
    incident_params = {}
    photoionisation_params = {}
    fixed_params = {}

    # Get incident grid axes and their ranges
    incident_axes = set(incident_grid.axes)

    # Automatically add all incident grid axes with their full ranges
    for axis in incident_grid.axes:
        axis_values = getattr(incident_grid, axis)
        min_val = float(np.min(axis_values))
        max_val = float(np.max(axis_values))
        # Default to log scale for ages and metallicities
        scale = "log" if axis in ["ages", "age", "metallicities", "metallicity"] else "linear"
        incident_params[axis] = (min_val, max_val, scale)

    for key, value in params.items():
        # Check if this is an incident grid parameter
        # Handle both singular (age, metallicity) and plural (ages, metallicities)
        is_incident = False
        incident_key = None

        if key in ["age", "ages"] and ("ages" in incident_axes or "age" in incident_axes):
            is_incident = True
            incident_key = "ages" if "ages" in incident_axes else "age"
        elif key in ["metallicity", "metallicities"] and (
            "metallicities" in incident_axes or "metallicity" in incident_axes
        ):
            is_incident = True
            incident_key = (
                "metallicities" if "metallicities" in incident_axes else "metallicity"
            )

        # Parse parameter value
        if isinstance(value, list) and len(value) == 3:
            min_val, max_val, scale = value

            if is_incident:
                # Override default with user-specified range
                incident_params[incident_key] = (min_val, max_val, scale)
            else:
                photoionisation_params[key] = (min_val, max_val, scale)

        # Handle nested abundance_scalings
        elif key == "abundance_scalings" and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list) and len(sub_value) == 3:
                    min_val, max_val, scale = sub_value
                    photoionisation_params[f"abundance_scalings.{sub_key}"] = (
                        min_val,
                        max_val,
                        scale,
                    )
                else:
                    if "abundance_scalings" not in fixed_params:
                        fixed_params["abundance_scalings"] = {}
                    fixed_params["abundance_scalings"][sub_key] = sub_value
        else:
            fixed_params[key] = value

    return incident_params, photoionisation_params, fixed_params


def generate_sobol_samples(incident_params, photoionisation_params, n_samples, seed=None):
    """
    Generate Sobol sequence samples for all parameters.

    Args:
        incident_params (dict): Incident grid parameters with (min, max, scale)
        photoionisation_params (dict): Photoionization parameters with (min, max, scale)
        n_samples (int): Number of samples
        seed (int, optional): Random seed

    Returns:
        tuple: (incident_samples, photoionisation_samples)
            Each is a list of dicts containing sampled parameter values
    """
    # Combine all parameters
    all_params = {**incident_params, **photoionisation_params}
    n_dims = len(all_params)
    param_names = list(all_params.keys())

    # Generate Sobol samples - use power of 2 for optimal coverage
    n_sobol = int(2 ** np.ceil(np.log2(n_samples)))
    sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
    unit_samples = sampler.random(n=n_sobol)[:n_samples]

    # Transform to parameter ranges
    samples = []
    for i in range(n_samples):
        sample = {}
        for j, param_name in enumerate(param_names):
            min_val, max_val, scale = all_params[param_name]

            if scale == "log":
                log_min = np.log10(min_val)
                log_max = np.log10(max_val)
                value = 10 ** (log_min + unit_samples[i, j] * (log_max - log_min))
            elif scale == "linear":
                value = min_val + unit_samples[i, j] * (max_val - min_val)
            else:
                raise ValueError(f"Unknown scale: {scale}")

            sample[param_name] = value
        samples.append(sample)

    # Split into incident and photoionization samples
    incident_param_names = set(incident_params.keys())
    incident_samples = [
        {k: v for k, v in s.items() if k in incident_param_names} for s in samples
    ]
    photoionisation_samples = [
        {k: v for k, v in s.items() if k not in incident_param_names} for s in samples
    ]

    return incident_samples, photoionisation_samples


if __name__ == "__main__":
    parser = Parser(
        description="Create Cloudy inputs with Sobol-sampled parameters",
        cloudy_args=True,
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Number of Sobol samples",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="Random seed for Sobol sampling",
    )

    parser.add_argument("--machine", type=str, required=False, default=None)

    parser.add_argument(
        "--cloudy-executable-path", type=str, required=False, default=None
    )

    parser.add_argument(
        "--no-striding",
        action="store_true",
        help="Disable striding in array jobs (use direct 1:1 mapping)",
    )

    parser.add_argument(
        "--stride-step",
        type=int,
        required=False,
        default=1000,
        help="Stride step for array jobs (default: 1000)",
    )

    parser.add_argument(
        "--time-per-model",
        type=int,
        required=False,
        default=10,
        help="Expected time per Cloudy model in minutes (default: 10)",
    )

    args = parser.parse_args()

    incident_grid_name = args.incident_grid
    incident_grid_dir = args.grid_dir
    cloudy_output_dir = args.cloudy_output_dir
    cloudy_paramfile = args.cloudy_paramfile
    n_samples = args.n_samples
    seed = args.seed
    machine = args.machine
    cloudy_executable_path = args.cloudy_executable_path

    print(f"Incident grid: {incident_grid_name}")
    print(f"Parameter file: {cloudy_paramfile}")
    print(f"Number of Sobol samples: {n_samples}")

    # Add extensions if needed
    if not cloudy_paramfile.endswith(".yaml"):
        cloudy_paramfile += ".yaml"
    if not incident_grid_name.endswith(".hdf5"):
        incident_grid_name += ".hdf5"

    # Generate grid name
    paramfile_base = Path(cloudy_paramfile).stem
    grid_base = Path(incident_grid_name).stem
    new_grid_name = f"{grid_base}_cloudy-sobol-{paramfile_base}-n{n_samples}"
    if seed is not None:
        new_grid_name += f"-seed{seed}"

    # Load parameter file
    with open(cloudy_paramfile, "r") as file:
        all_params = yaml.safe_load(file)

    # Load incident grid
    incident_grid = Grid(
        incident_grid_name,
        grid_dir=incident_grid_dir,
        ignore_lines=True,
    )

    # Parse parameters
    incident_params, photoionisation_params, fixed_params = parse_sobol_params(
        all_params, incident_grid
    )

    print(f"\nIncident parameters: {list(incident_params.keys())}")
    print(f"Photoionization parameters: {list(photoionisation_params.keys())}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")

    # Generate Sobol samples
    incident_samples, photoionisation_samples = generate_sobol_samples(
        incident_params, photoionisation_params, n_samples, seed=seed
    )

    # Create output directory
    output_directory = f"{cloudy_output_dir}/{new_grid_name}"
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Determine Cloudy version
    cloudy_version = fixed_params["cloudy_version"]
    if cloudy_version == "c17.03":
        cloudy = cloudy17
    elif cloudy_version == "c23.01":
        cloudy = cloudy23
    elif cloudy_version == "c25.00":
        cloudy = cloudy23  # Use cloudy23 module for c25.00 as well
    else:
        raise ValueError(f"Unknown Cloudy version: {cloudy_version}")

    # Save grid parameters
    parameters_to_save = {
        "incident_n_models": int(n_samples),
        "photoionisation_n_models": 1,  # Single photoionization model per incident
        "total_n_models": int(n_samples),
        "sampling_method": "sobol",
        "seed": seed,
        **fixed_params,
        "incident_params": {k: list(v) for k, v in incident_params.items()},
        "photoionisation_params": {
            k: list(v) for k, v in photoionisation_params.items()
        },
        "depletion_model": None,
    }

    with open(f"{output_directory}/grid_parameters.yaml", "w") as file:
        yaml.dump(parameters_to_save, file, default_flow_style=False)

    # Save the actual sampled parameter values to HDF5
    grid_hdf5_file = f"{output_directory}/{new_grid_name}.hdf5"
    with h5py.File(grid_hdf5_file, "w") as hf:
        # Create parameters group
        param_group = hf.create_group("parameters")

        # Save sampled parameter values for each model
        for i, (inc_sample, photo_sample) in enumerate(
            zip(incident_samples, photoionisation_samples)
        ):
            # Combine incident and photoionisation samples
            sample_params = {**inc_sample, **photo_sample}
            for key, value in sample_params.items():
                if key not in param_group:
                    # Create dataset on first encounter
                    param_group.create_dataset(
                        key, data=np.zeros(n_samples), compression="gzip"
                    )
                param_group[key][i] = float(value)

        # Save fixed parameters as attributes
        for key, value in fixed_params.items():
            if value is None or (isinstance(value, dict) and not value):
                continue
            if isinstance(value, dict):
                import json

                hf.attrs[key] = json.dumps(value)
            elif isinstance(value, (list, tuple)):
                hf.attrs[key] = np.array(value)
            else:
                hf.attrs[key] = value

        # Save grid metadata
        hf.attrs["n_samples"] = n_samples
        hf.attrs["sampling_method"] = "sobol"
        if seed is not None:
            hf.attrs["seed"] = seed

    print(f"Saved parameter grid to {grid_hdf5_file}")

    shutil.copy(cloudy_paramfile, f"{output_directory}/")

    # Get wavelength array
    lam = incident_grid.lam

    # Extract sampled ages and metallicities for Stars object
    ages_array = np.array([s.get("ages", s.get("age", None)) for s in incident_samples])
    metallicities_array = np.array(
        [s.get("metallicities", s.get("metallicity", None)) for s in incident_samples]
    )

    # Create Stars object for spectral interpolation
    initial_masses = np.ones(n_samples) * Msun
    stars = Stars(
        initial_masses=initial_masses,
        ages=ages_array * yr,  # Convert log10(age/Gyr) to linear age
        metallicities=metallicities_array,
    )

    # Get interpolated spectra
    emodel = IncidentEmission(incident_grid, per_particle=True)
    print("\nGenerating interpolated spectra...")
    spec = stars.get_spectra(emodel, nthreads=-1)

    with h5py.File(grid_hdf5_file, "a") as hf:
        hf.create_dataset("incident_lam", data=lam, compression="gzip")
        hf.create_dataset("incident_lnu", data=spec.lnu.value, compression="gzip")

    # Handle reference ionisation parameter if needed
    if fixed_params.get("ionisation_parameter_model") == "ref":
        # Get reference point parameters
        ref_age = fixed_params.get("reference_age")
        ref_met = fixed_params.get("reference_metallicity")

        if ref_age is None or ref_met is None:
            raise ValueError(
                "reference_age and reference_metallicity required for 'ref' model"
            )

        # Get reference grid point
        ref_dict = {}
        for axis in incident_grid.axes:
            if axis in ["ages", "age"]:
                ref_dict[axis] = ref_age
            elif axis in ["metallicities", "metallicity"]:
                ref_dict[axis] = ref_met

        ref_grid_point = incident_grid.get_grid_point(**ref_dict)
        reference_log10_specific_ionising_lum = (
            incident_grid.log10_specific_ionising_lum["HI"][ref_grid_point]
        )

        # Calculate specific ionising luminosity for each sample
        # Need to interpolate log10_specific_ionising_lum
        ref_stars = Stars(
            initial_masses=initial_masses,
            ages=10**ages_array * yr,
            metallicities=metallicities_array,
        )
        # Get ionising photon luminosity at sampled points
        log10Q_samples = np.log10(
            ref_stars.get_attr_per_particle(
                incident_grid, "log10_specific_ionising_lum", "HI"
            )
        )
    else:
        reference_log10_specific_ionising_lum = None
        log10Q_samples = None

    # Save ionisation parameter scaling if needed (for on-the-fly generation)
    if fixed_params.get("ionisation_parameter_model") == "ref":
        with h5py.File(grid_hdf5_file, "a") as hf:
            hf.attrs["reference_log10_specific_ionising_lum"] = reference_log10_specific_ionising_lum
            hf.create_dataset(
                "log10Q_samples",
                data=log10Q_samples,
                compression="gzip",
            )

    linelist_name = fixed_params.get("output_linelist", "linelist.dat")
    shutil.copyfile(linelist_name, f"{output_directory}/{linelist_name}")

    print(f"\nSaved parameter grid and incident spectra to {grid_hdf5_file}")
    print(f"Cloudy input files will be generated on-the-fly during execution")
    print(f"Output directory: {output_directory}")

    # Generate submission script (on-the-fly version)
    if machine:
        if machine == "cosma7":
            submission_scripts.cosma7_sobol_onthefly(
                new_grid_name=new_grid_name,
                n_samples=n_samples,
                cloudy_output_dir=cloudy_output_dir,
                cloudy_executable_path=cloudy_executable_path,
                time_per_model=args.time_per_model,
                use_striding=not args.no_striding,
                stride_step=args.stride_step,
            )
        else:
            getattr(submission_scripts, machine)(
                new_grid_name,
                number_of_incident_grid_points=n_samples,
                number_of_photoionisation_models=1,
                cloudy_output_dir=cloudy_output_dir,
                cloudy_executable_path=cloudy_executable_path,
                memory="4G",
                by_photoionisation_grid_point=False,
            )
