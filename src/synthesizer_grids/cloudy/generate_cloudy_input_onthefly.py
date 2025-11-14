"""
Generate a single Cloudy input file on-the-fly from the parameter grid.

This script reads the HDF5 parameter grid and creates Cloudy input files
for a specific sample index without requiring all files to exist on disk.
"""

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml
from synthesizer.photoionisation import cloudy17, cloudy23
from unyt import Angstrom, erg, Hz, s

from create_cloudy_input_grid import create_cloudy_input


def generate_input_for_index(grid_dir, sample_index, work_dir=None):
    """
    Generate Cloudy input file for a specific sample index.

    Args:
        grid_dir (str): Directory containing the HDF5 grid file
        sample_index (int): Index of the sample to generate
        work_dir (str): Working directory for temporary files (default: {grid_dir}/{sample_index})

    Returns:
        str: Path to the working directory
    """
    grid_dir = Path(grid_dir)

    # Find HDF5 file
    hdf5_files = list(grid_dir.glob("*.hdf5"))
    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No HDF5 file found in {grid_dir}")
    elif len(hdf5_files) > 1:
        raise ValueError(f"Multiple HDF5 files found in {grid_dir}")

    grid_file = hdf5_files[0]
    grid_name = grid_dir.name

    # Load grid parameters
    param_file = grid_dir / "grid_parameters.yaml"
    with open(param_file, "r") as f:
        grid_params = yaml.safe_load(f)

    cloudy_version = grid_params["cloudy_version"]
    if cloudy_version.split(".")[0] in ["c23", "c25"]:
        cloudy = cloudy23
    elif cloudy_version.split(".")[0] == "c17":
        cloudy = cloudy17
    else:
        raise ValueError(f"Unknown Cloudy version: {cloudy_version}")

    # Read data from HDF5
    with h5py.File(grid_file, "r") as hf:
        n_samples = hf.attrs["n_samples"]

        if sample_index >= n_samples or sample_index < 0:
            raise ValueError(
                f"Sample index {sample_index} out of range [0, {n_samples-1}]"
            )

        # Get parameters for this sample
        sample_params = {}
        for key in hf["parameters"].keys():
            sample_params[key] = float(hf["parameters"][key][sample_index])

        # Get incident spectrum
        lam = hf["incident_lam"][:] * Angstrom
        lnu = hf["incident_lnu"][sample_index, :] * erg / s / Hz

        # Get fixed parameters from attributes
        fixed_params = {}
        for key in hf.attrs.keys():
            if key not in ["n_samples", "sampling_method", "seed", "n_wavelength", "spec_names"]:
                value = hf.attrs[key]
                if isinstance(value, bytes):
                    import json
                    fixed_params[key] = json.loads(value.decode())
                else:
                    fixed_params[key] = value

        # Get ionisation parameter scaling if exists
        delta_log10_specific_ionising_luminosity = None
        if "log10Q_samples" in hf:
            reference_log10_specific_ionising_lum = hf.attrs[
                "reference_log10_specific_ionising_lum"
            ]
            log10Q_sample = hf["log10Q_samples"][sample_index]
            delta_log10_specific_ionising_luminosity = (
                log10Q_sample - reference_log10_specific_ionising_lum
            )

    # Create working directory
    if work_dir is None:
        work_dir = grid_dir / str(sample_index)
    else:
        work_dir = Path(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)

    # Save incident SED
    np.save(f"{work_dir}/input", np.array([lam.value, lnu.value]))

    # Convert to Cloudy format
    cloudy.ShapeCommands.table_sed(
        "input",
        lam,
        lnu,
        output_dir=work_dir,
    )

    # Combine all parameters
    parameters = {**sample_params, **fixed_params}

    # Create Cloudy input file
    # create_cloudy_input expects {output_directory}/{incident_index}/{photoionisation_index}.in
    # We want files in work_dir, so structure accordingly
    temp_parent = work_dir.parent
    temp_incident_name = work_dir.name

    # Create incident directory if it doesn't exist
    (temp_parent / temp_incident_name).mkdir(parents=True, exist_ok=True)

    linelist_name = parameters.get("output_linelist", "linelist.dat")
    shutil.copyfile(grid_dir / linelist_name, work_dir / linelist_name)

    # Change to work_dir so create_cloudy_input can find linelist
    import os
    original_dir = os.getcwd()
    os.chdir(temp_parent)

    try:
        create_cloudy_input(
            incident_index=temp_incident_name,
            photoionisation_index=0,
            parameters=parameters,
            delta_log10_specific_ionising_luminosity=delta_log10_specific_ionising_luminosity,
            output_directory=str(temp_parent),
        )
    finally:
        os.chdir(original_dir)

    return str(work_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Cloudy input file for a specific sample index"
    )

    parser.add_argument(
        "--grid-dir",
        type=str,
        required=True,
        help="Directory containing the HDF5 grid file",
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        required=True,
        help="Index of the sample to generate",
    )

    parser.add_argument(
        "--work-dir",
        type=str,
        required=False,
        default=None,
        help="Working directory for temporary files (default: {grid_dir}/{sample_index})",
    )

    args = parser.parse_args()

    work_dir = generate_input_for_index(args.grid_dir, args.sample_index, args.work_dir)
    print(f"Created Cloudy input files in {work_dir}")
