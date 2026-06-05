"""
Collect Cloudy output continuum spectra from Sobol-sampled models and add
to the HDF5 grid created by create_cloudy_input_sobol.py.

Reads the raw cloudy ``.cont`` files from the ``cont/`` subdirectory (one per
sample, named ``{index}.cont``) using synthesizer's ``read_continuum`` reader,
which fixes the column mapping, wavelength ordering and units to match the
stored incident spectra.

Usage:
    python collect_sobol_outputs.py --output-dir /path/to/cloudy_output_dir
    python collect_sobol_outputs.py --output-dir /path/to/dir \
        --spec-names nebular linecont
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

from synthesizer_grids.cloudy import cloudy17, cloudy23, cloudy25

# All spectra exposed by synthesizer.read_continuum(return_dict=True).
AVAILABLE_SPECTRA = (
    "incident",
    "transmitted",
    "nebular",
    "nebular_continuum",
    "total",
    "linecont",
)

# Base spectra to store; total and nebular_continuum are derived from these
# within synthesizer and so are not saved by default.
DEFAULT_SPECTRA = (
    "incident",
    "transmitted",
    "nebular",
    "linecont",
)


def _select_cloudy_module(cloudy_version):
    """Return the synthesizer cloudy module for a given version string."""
    major = cloudy_version.split(".")[0]
    if major == "c23":
        return cloudy23
    elif major == "c25":
        return cloudy25
    elif major == "c17":
        return cloudy17
    raise ValueError(f"Unknown Cloudy version: {cloudy_version}")


def collect_sobol_outputs(
    output_dir, output_file=None, spec_names=DEFAULT_SPECTRA
):
    """
    Collect all Sobol output spectra into the parameter HDF5.

    Args:
        output_dir (str): Path to Cloudy output directory
        output_file (str): Output HDF5 filename (optional)
        spec_names (tuple): Spectra to save (any of AVAILABLE_SPECTRA).
    """
    output_dir = Path(output_dir)

    for spec_name in spec_names:
        if spec_name not in AVAILABLE_SPECTRA:
            raise ValueError(
                f"Unknown spectrum type: {spec_name}. "
                f"Choose from {AVAILABLE_SPECTRA}."
            )

    # Determine HDF5 grid file path. If not specified, find the single .hdf5
    # in the output dir (created by create_cloudy_input_sobol.py).
    if output_file is None:
        hdf5_files = sorted(output_dir.glob("*.hdf5"))
        if len(hdf5_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one .hdf5 in {output_dir}, found "
                f"{len(hdf5_files)}. Run create_cloudy_input_sobol.py first."
            )
        output_file = hdf5_files[0]
    else:
        output_file = Path(output_file)
        if not output_file.exists():
            raise FileNotFoundError(f"Grid HDF5 file not found: {output_file}")

    # Load grid parameters from YAML for cloudy version
    param_file = output_dir / "grid_parameters.yaml"
    with open(param_file, "r") as f:
        grid_params = yaml.safe_load(f)

    cloudy = _select_cloudy_module(grid_params["cloudy_version"])

    # Read n_samples from existing HDF5 file
    with h5py.File(output_file, "r") as hf:
        n_samples = int(hf.attrs["n_samples"])

    print(f"Collecting {n_samples} Sobol samples from {output_dir}")
    print(f"Cloudy version: {grid_params['cloudy_version']}")
    print(f"Spectra to save: {spec_names}")

    # The submission script saves the raw cloudy continuum for each sample as
    # cont/{index}.cont (synthesizer's read_continuum appends the .cont).
    cont_dir = output_dir / "cont"
    available = sorted(cont_dir.glob("*.cont"))
    if not available:
        raise FileNotFoundError(f"No .cont files found in {cont_dir}")

    # Read the (ascending, Angstrom) wavelength grid from the first file
    first_index = available[0].stem  # "{index}.cont" -> "{index}"
    lam = cloudy.read_wavelength(str(cont_dir / first_index))
    n_lambda = len(lam)
    print(f"Wavelength grid size: {n_lambda}")

    # Initialise arrays for requested spectra
    spectra = {
        spec_name: np.zeros((n_samples, n_lambda)) for spec_name in spec_names
    }

    # Collect spectra, recording any samples we cannot use
    missing = []  # no .cont file (failed/never-run cloudy model)
    bad = []  # .cont present but unreadable / wrong wavelength length
    for i in range(n_samples):
        cont_prefix = cont_dir / str(i)
        if not cont_prefix.with_suffix(".cont").exists():
            missing.append(i)
            continue

        try:
            spec_dict = cloudy.read_continuum(
                str(cont_prefix), return_dict=True
            )
        except Exception as exc:  # noqa: BLE001 - want to skip any bad file
            print(f"Warning: could not read sample {i}: {exc}")
            bad.append(i)
            continue

        if len(spec_dict["lam"]) != n_lambda:
            print(
                f"Warning: sample {i} has {len(spec_dict['lam'])} wavelength "
                f"points, expected {n_lambda}"
            )
            bad.append(i)
            continue

        for spec_name in spec_names:
            spectra[spec_name][i, :] = spec_dict[spec_name]

    # Build boolean mask: True only for samples with usable spectra. Both
    # missing files and unreadable/mismatched files are excluded so all-zero
    # rows never leak into downstream training as "valid".
    invalid = sorted(set(missing) | set(bad))
    valid_samples = np.ones(n_samples, dtype=bool)
    valid_samples[invalid] = False
    n_valid = int(valid_samples.sum())

    if missing:
        print(
            f"\n{len(missing)} missing .cont files (failed/absent runs): "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if bad:
        print(
            f"{len(bad)} unreadable/inconsistent .cont files: "
            f"{bad[:10]}{'...' if len(bad) > 10 else ''}"
        )
    print(f"\nValid samples: {n_valid}/{n_samples} ({len(invalid)} excluded)")

    print(f"\nAdding spectra to {output_file}")

    # Append spectra to existing HDF5 file
    # (parameters already saved by create_cloudy_input_sobol.py)
    with h5py.File(output_file, "a") as hf:
        # Create spectra group if it doesn't exist
        if "spectra" not in hf:
            spectra_group = hf.create_group("spectra")
        else:
            spectra_group = hf["spectra"]

        # Save requested spectra (allow re-running by overwriting)
        for spec_name in spec_names:
            if spec_name in spectra_group:
                del spectra_group[spec_name]
            spectra_group.create_dataset(
                spec_name, data=spectra[spec_name], compression="gzip"
            )

        # Save wavelength grid (Angstroms, ascending)
        if "wavelength" in hf:
            del hf["wavelength"]
        hf.create_dataset("wavelength", data=lam, compression="gzip")

        # Save valid samples mask
        if "valid_samples" in hf:
            del hf["valid_samples"]
        hf.create_dataset("valid_samples", data=valid_samples)

        # Update metadata
        hf.attrs["n_wavelength"] = n_lambda
        hf.attrs["n_valid_samples"] = n_valid
        hf.attrs["spec_names"] = list(spec_names)

    print(
        f"Successfully added spectra ({', '.join(spec_names)}) "
        f"to {output_file}"
    )


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
        help="HDF5 grid file path (default: auto-detected from "
        "output_dir name)",
    )

    parser.add_argument(
        "--spec-names",
        type=str,
        nargs="+",
        required=False,
        default=list(DEFAULT_SPECTRA),
        choices=list(AVAILABLE_SPECTRA),
        help="Spectra to save (default: incident transmitted nebular "
        "linecont).",
    )

    args = parser.parse_args()

    collect_sobol_outputs(
        args.output_dir, args.output_file, spec_names=tuple(args.spec_names)
    )
