"""
Collect Cloudy output continuum spectra from Sobol-sampled models and add to HDF5 grid.

Reads spectra from spectra/*.txt files and adds to the HDF5 file created by
create_cloudy_input_sobol.py. The HDF5 file already contains:
- Sampled parameter values for each model (in parameters/ group)
- Fixed parameters as attributes

This script adds:
- Continuum spectra as 2D dataset(s) - choose from total, nebular, transmitted
- Wavelength grid

Usage:
    # Save total spectrum only (default)
    python collect_sobol_outputs.py \
        --output-dir /path/to/cloudy_output_dir

    # Save multiple spectra
    python collect_sobol_outputs.py \
        --output-dir /path/to/cloudy_output_dir \
        --spec-names total nebular transmitted
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml


def collect_sobol_outputs(output_dir, output_file=None, spec_names=("total",)):
    """
    Collect all Sobol output spectra and parameters into HDF5.

    Args:
        output_dir (str): Path to Cloudy output directory
        output_file (str): Output HDF5 filename (optional)
        spec_names (tuple): Spectra to save - choose from "total", "nebular", "transmitted"
                           (default: ("total",))
    """
    output_dir = Path(output_dir)

    # Determine HDF5 grid file path
    # If output_file not specified, use the grid name from the directory
    if output_file is None:
        grid_name = output_dir.name
        output_file = output_dir / f"{grid_name}.hdf5"
    else:
        output_file = Path(output_file)

    # Check if HDF5 file exists (created by create_cloudy_input_sobol.py)
    if not output_file.exists():
        raise FileNotFoundError(
            f"Grid HDF5 file not found: {output_file}. "
            f"Run create_cloudy_input_sobol.py first to create the parameter grid."
        )

    # Load grid parameters from YAML for cloudy version
    param_file = output_dir / "grid_parameters.yaml"
    with open(param_file, "r") as f:
        grid_params = yaml.safe_load(f)

    cloudy_version = grid_params["cloudy_version"]

    # Read n_samples and parameters from existing HDF5 file
    with h5py.File(output_file, "r") as hf:
        n_samples = hf.attrs["n_samples"]
        # Get parameter names from the parameters group
        varying_param_names = list(hf["parameters"].keys())

    print(f"Collecting {n_samples} Sobol samples from {output_dir}")
    print(f"Cloudy version: {cloudy_version}")

    # Read wavelength grid from first spectra file
    first_spec_file = output_dir / "spectra" / "spectra_0.txt"
    if not first_spec_file.exists():
        raise FileNotFoundError(f"No spectra files found in {output_dir / 'spectra'}")

    first_data = np.loadtxt(first_spec_file)
    nu = first_data[:, 0]
    n_lambda = len(nu)

    # Convert frequency to wavelength (Angstrom)
    c_angstrom_hz = 2.99792458e18
    lam = c_angstrom_hz / nu

    print(f"Wavelength grid size: {n_lambda}")
    print(f"Spectra to save: {spec_names}")

    # Initialize arrays for requested spectra
    spectra = {}
    for spec_name in spec_names:
        spectra[spec_name] = np.zeros((n_samples, n_lambda))

    # Collect spectra from indexed text files in spectra/ subdirectory
    missing_files = []
    for i in range(n_samples):
        spec_file = output_dir / "spectra" / f"spectra_{i}.txt"

        if not spec_file.exists():
            missing_files.append(i)
            continue

        # Read spectrum from text file (columns: nu, transmitted, nebular, total)
        data = np.loadtxt(spec_file)
        nu_sample = data[:, 0]

        # Verify wavelength grid consistency
        if len(nu_sample) != n_lambda:
            print(f"Warning: Sample {i} has {len(nu_sample)} wavelength points, expected {n_lambda}")
            continue

        # Store requested spectra
        for spec_name in spec_names:
            if spec_name == "transmitted":
                spectra["transmitted"][i, :] = data[:, 1]
            elif spec_name == "nebular":
                spectra["nebular"][i, :] = data[:, 2]
            elif spec_name == "total":
                spectra["total"][i, :] = data[:, 3]
            else:
                raise ValueError(f"Unknown spectrum type: {spec_name}")

    if missing_files:
        print(f"\nWarning: {len(missing_files)} missing spectra files: {missing_files[:10]}...")

    print(f"\nAdding spectra to {output_file}")

    # Append spectra to existing HDF5 file (parameters already saved by create_cloudy_input_sobol.py)
    with h5py.File(output_file, "a") as hf:
        # Create spectra group if it doesn't exist
        if "spectra" not in hf:
            spectra_group = hf.create_group("spectra")
        else:
            spectra_group = hf["spectra"]

        # Save requested spectra
        for spec_name in spec_names:
            # If single spectrum, save directly under spectra group
            if len(spec_names) == 1:
                dataset_name = spec_name
            else:
                dataset_name = spec_name

            # Delete if already exists (allow re-running)
            if dataset_name in spectra_group:
                del spectra_group[dataset_name]

            spectra_group.create_dataset(
                dataset_name, data=spectra[spec_name], compression="gzip"
            )

        # Save wavelength grid (already in Angstroms from cloudy module)
        if "wavelength" in hf:
            del hf["wavelength"]
        hf.create_dataset("wavelength", data=lam, compression="gzip")

        # Update metadata
        hf.attrs["n_wavelength"] = n_lambda
        hf.attrs["spec_names"] = list(spec_names)

    print(f"Successfully added {n_samples} spectra ({', '.join(spec_names)}) to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect Cloudy Sobol outputs into HDF5"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to Cloudy output directory",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default=None,
        help="HDF5 grid file path (default: auto-detected from output_dir name)",
    )

    parser.add_argument(
        "--spec-names",
        type=str,
        nargs="+",
        required=False,
        default=["total", "transmitted", "nebular"],
        choices=["total", "nebular", "transmitted"],
        help="Spectra to save (default: total transmitted nebular). Can specify multiple: --spec-names total nebular transmitted",
    )

    args = parser.parse_args()

    collect_sobol_outputs(
        args.output_dir, args.output_file, spec_names=tuple(args.spec_names)
    )
