"""Append Cloudy outputs from a Sobol run into its HDF5 file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import h5py
import numpy as np
import yaml
from synthesizer.emissions import Sed
from unyt import Angstrom, Hz, erg, eV, s

from synthesizer_grids.cloudy import cloudy17, cloudy23
from synthesizer_grids.cloudy.create_synthesizer_grid import (
    CloudyOutputLocator,
    check_if_failed,
)

VALID_SPECTRA = ("incident", "transmitted", "nebular", "linecont", "total")


def _select_cloudy_module(version: str):
    prefix = version.split(".")[0].lower()
    if prefix in ("c23", "c25"):
        return cloudy23
    if prefix == "c17":
        return cloudy17
    raise ValueError(f"Unsupported Cloudy version '{version}'")


def _resolve_grid_file(
    run_dir: Path, metadata: Dict[str, object], override: str | None
) -> Path:
    if override:
        return Path(override)
    if "grid_hdf5" in metadata:
        return Path(metadata["grid_hdf5"])
    return run_dir / f"{run_dir.name}.hdf5"


def _initialise_line_arrays(n_samples: int, locator: CloudyOutputLocator):
    with open(locator.linelist_path, "r") as handle:
        raw_lines = handle.readlines()
    clean_lines = [line.strip() for line in raw_lines if line.strip()]
    n_lines = len(clean_lines)
    arrays = {
        "luminosity": np.zeros((n_samples, n_lines)),
        "transmitted": np.zeros((n_samples, n_lines)),
        "incident": np.zeros((n_samples, n_lines)),
        "nebular_continuum": np.zeros((n_samples, n_lines)),
        "total_continuum": np.zeros((n_samples, n_lines)),
    }
    return n_lines, arrays


def _extract_spectrum(
    spec_dict: Dict[str, np.ndarray], name: str
) -> np.ndarray:
    if name == "total" and "total" not in spec_dict:
        return spec_dict["transmitted"] + spec_dict["nebular"]
    return spec_dict[name]


def collect_sobol_outputs(
    output_dir: str,
    output_file: str | None = None,
    spec_names: Iterable[str] = (
        "incident",
        "transmitted",
        "nebular",
        "linecont",
    ),
    include_lines: bool = True,
):
    run_dir = Path(output_dir)
    metadata_path = run_dir / "grid_parameters.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    with open(metadata_path, "r") as handle:
        metadata = yaml.safe_load(handle)

    grid_file = _resolve_grid_file(run_dir, metadata, output_file)
    if not grid_file.exists():
        raise FileNotFoundError(
            f"Sobol grid file {grid_file} does not exist; "
            "run the workflow first"
        )

    locator = CloudyOutputLocator(
        run_dir.parent,
        run_dir.name,
        photoionisation_n_models=metadata.get("photoionisation_n_models"),
    )
    cloudy_module = _select_cloudy_module(metadata["cloudy_version"])
    lam = cloudy_module.read_wavelength(str(locator.model_base_path(0, 0)))

    with h5py.File(grid_file, "r") as hf:
        n_samples = int(hf.attrs["n_samples"])
        log10q_samples = (
            hf["log10Q_samples"][:] if "log10Q_samples" in hf else None
        )
        reference_log10q = hf.attrs.get(
            "reference_log10_specific_ionising_lum"
        )

    spectra_arrays = {
        name: np.zeros((n_samples, len(lam))) for name in spec_names
    }
    normalisations = np.ones(n_samples)
    failures = np.zeros(n_samples, dtype=int)

    if include_lines:
        _, line_arrays = _initialise_line_arrays(n_samples, locator)
        line_wavelengths = None
        line_ids = None
    else:
        line_arrays = {}
        line_wavelengths = None
        line_ids = None

    missing = []
    for sample_idx in range(n_samples):
        failed = check_if_failed(
            locator,
            sample_idx,
            0,
        )
        if failed:
            failures[sample_idx] = 1
            missing.append(sample_idx)
            continue

        base_path = str(locator.model_base_path(sample_idx, 0))
        spec_dict = cloudy_module.read_continuum(base_path, return_dict=True)

        normalisation = 1.0
        if reference_log10q is not None and log10q_samples is not None:
            sed = Sed(
                lam=lam * Angstrom,
                lnu=spec_dict["incident"] * erg / s / Hz,
            )
            ionising_photon_production_rate = (
                sed.calculate_ionising_photon_production_rate(
                    ionisation_energy=13.6 * eV,
                    limit=100,
                )
            )
            normalisation = 10 ** (
                log10q_samples[sample_idx]
                - np.log10(ionising_photon_production_rate)
            )
        normalisations[sample_idx] = normalisation

        for name in spec_names:
            spectra_arrays[name][sample_idx] = (
                _extract_spectrum(spec_dict, name) * normalisation
            )

        if include_lines:
            ids, wavelengths, luminosities = cloudy_module.read_linelist(
                base_path,
                extension="emergent_elin",
            )
            order = np.argsort(wavelengths)
            wavelengths = wavelengths[order]
            luminosities = luminosities[order] * normalisation
            ids = ids[order]

            if line_wavelengths is None:
                line_wavelengths = wavelengths * Angstrom
                line_ids = ids

            line_arrays["luminosity"][sample_idx] = luminosities

            transmitted = spec_dict["transmitted"] * normalisation
            incident = spec_dict["incident"] * normalisation
            nebular_continuum = (
                spec_dict["nebular"] - spec_dict["linecont"]
            ) * normalisation
            total_continuum = transmitted + nebular_continuum

            line_arrays["transmitted"][sample_idx] = np.interp(
                wavelengths,
                lam,
                transmitted,
            )
            line_arrays["incident"][sample_idx] = np.interp(
                wavelengths,
                lam,
                incident,
            )
            line_arrays["nebular_continuum"][sample_idx] = np.interp(
                wavelengths,
                lam,
                nebular_continuum,
            )
            line_arrays["total_continuum"][sample_idx] = np.interp(
                wavelengths,
                lam,
                total_continuum,
            )

    if missing:
        print(
            f"Warning: {len(missing)} models missing outputs: {missing[:10]}"
        )

    with h5py.File(grid_file, "a") as hf:
        spectra_group = hf.require_group("spectra")
        for name, data in spectra_arrays.items():
            if name in spectra_group:
                del spectra_group[name]
            dataset = spectra_group.create_dataset(
                name,
                data=data,
                compression="gzip",
            )
            dataset.attrs["Units"] = "erg/s/Hz"

        if "wavelength" in hf:
            del hf["wavelength"]
        wave = hf.create_dataset("wavelength", data=lam, compression="gzip")
        wave.attrs["Units"] = "Angstrom"

        if "normalisation" in hf:
            del hf["normalisation"]
        hf.create_dataset("normalisation", data=normalisations)

        if "failures" in hf:
            del hf["failures"]
        hf.create_dataset("failures", data=failures)

        if include_lines and line_wavelengths is not None:
            lines_group = hf.require_group("lines")
            for key, data in line_arrays.items():
                if key in lines_group:
                    del lines_group[key]
                dataset = lines_group.create_dataset(
                    key,
                    data=data,
                    compression="gzip",
                )
                if key == "luminosity":
                    dataset.attrs["Units"] = "erg/s"
                else:
                    dataset.attrs["Units"] = "erg/s/Hz"
            if "wavelength" in lines_group:
                del lines_group["wavelength"]
            wl = lines_group.create_dataset(
                "wavelength",
                data=line_wavelengths.value,
                compression="gzip",
            )
            wl.attrs["Units"] = "Angstrom"
            if line_ids is not None:
                if "id" in lines_group:
                    del lines_group["id"]
                lines_group.create_dataset("id", data=line_ids.astype("S"))

    print("Added spectra" f" ({', '.join(spec_names)}) to {grid_file}.")
    if include_lines:
        print("Added line luminosities and continuum estimates.")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Sobol Cloudy outputs into an existing HDF5 file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the Sobol Cloudy run directory",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional explicit path to the Sobol HDF5 file",
    )
    parser.add_argument(
        "--spec-names",
        nargs="+",
        choices=VALID_SPECTRA,
        default=["incident", "transmitted", "nebular", "linecont"],
        help="Spectra to store in the HDF5 file",
    )
    parser.add_argument(
        "--no-lines",
        action="store_true",
        help="Skip ingesting emission line outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    collect_sobol_outputs(
        args.output_dir,
        output_file=args.output_file,
        spec_names=tuple(args.spec_names),
        include_lines=not args.no_lines,
    )
