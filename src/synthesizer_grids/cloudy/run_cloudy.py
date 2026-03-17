"""On-the-fly Cloudy execution helper."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import yaml
from synthesizer.grid import Grid
from unyt import Angstrom, Hz, erg, s

from synthesizer_grids.cloudy import cloudy17, cloudy23
from synthesizer_grids.cloudy.input_writer import CloudyInputWriter
from synthesizer_grids.cloudy.utils import get_grid_props_cloudy


def _select_cloudy_module(version: str):
    version = version.lower()
    if version.startswith("c17"):
        return cloudy17
    return cloudy23


def _load_metadata(path: Path) -> Dict[str, object]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _copy_linelist_if_needed(source: Path, destination_dir: Path) -> Path:
    destination = destination_dir / source.name
    if not destination.exists():
        if not source.exists():
            raise FileNotFoundError(f"Linelist not found: {source}")
        shutil.copy2(source, destination)
    return destination


def _cloudy_run_succeeded(incident_dir: Path, photo_index: int) -> bool:
    out_file = incident_dir / f"{photo_index}.out"
    if not out_file.exists():
        return False
    try:
        with open(out_file, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - 8192, 0))
            data = handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return False
    lines = [line.strip() for line in data.splitlines() if line.strip()]
    if not lines:
        return False
    success_token = "Cloudy exited OK"
    tail_window = lines[-20:]
    return any(success_token in line for line in tail_window)


def _store_outputs(
    incident_dir: Path,
    output_dir: Path,
    incident_index: int,
    photo_index: int,
    photo_n: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = (
        f"{incident_index}"
        if photo_n == 1
        else f"{incident_index}_{photo_index}"
    )
    artifacts = {
        incident_dir / f"{photo_index}.cont": output_dir / f"{base}.cont",
        incident_dir / f"{photo_index}.emergent_elin": output_dir
        / f"{base}.emergent_elin",
    }
    for source, target in artifacts.items():
        if not source.exists():
            continue
        if target.exists():
            target.unlink()
        shutil.move(source, target)


class IncidentSampler:
    def __init__(self, metadata: Dict[str, object]):
        self.metadata = metadata
        grid_path = Path(metadata["incident_grid_path"])
        self.grid = Grid(
            grid_path.name,
            grid_dir=str(grid_path.parent),
            ignore_lines=True,
        )
        self.incident_axes: List[str] = metadata.get(
            "incident_axes", self.grid.axes
        )
        axis_values_meta = metadata.get("incident_axes_values", {})
        self.incident_axes_values: Dict[str, np.ndarray] = {}
        for axis in self.incident_axes:
            values = axis_values_meta.get(axis)
            if values is None:
                values = self.grid.axes_values[axis]
            self.incident_axes_values[axis] = np.array(values)
        self.incident_shape = tuple(
            len(self.incident_axes_values[axis]) for axis in self.incident_axes
        )

        self.fixed_params = metadata.get("fixed_parameters", {}).copy()
        variable = metadata.get("variable_parameters", {}) or {}
        self.variable_params = {
            key: np.array(value) for key, value in variable.items()
        }
        if self.variable_params:
            photo_axes = list(self.variable_params.keys())
            (_, _, _, _, photo_model_list, _) = get_grid_props_cloudy(
                photo_axes, self.variable_params, verbose=False
            )
            self.photo_parameter_grid = [
                {axis: model[idx] for idx, axis in enumerate(photo_axes)}
                for model in photo_model_list
            ]
        else:
            self.photo_parameter_grid = [{}]

        self.reference_log10 = self._compute_reference_log10()

        linelist_path = metadata.get("linelist_path")
        self.linelist_path = Path(linelist_path) if linelist_path else None
        self.linelist_name = metadata.get("linelist_name", "linelist.dat")

    def _compute_reference_log10(self) -> float | None:
        if self.fixed_params.get("ionisation_parameter_model") != "ref":
            return None
        reference_values = []
        for axis in self.incident_axes:
            lookup = (
                "age"
                if axis == "ages"
                else "metallicity"
                if axis == "metallicities"
                else axis
            )
            ref_key = f"reference_{lookup}"
            if ref_key not in self.fixed_params:
                raise ValueError(
                    f"Missing {ref_key} in fixed parameters "
                    "for ref ionisation model"
                )
            reference_values.append(self.fixed_params[ref_key])
        ref_dict = dict(zip(self.incident_axes, reference_values))
        ref_point = self.grid.get_grid_point(**ref_dict)
        return self.grid.log10_specific_ionising_lum["HI"][ref_point]

    def prepare_model(self, incident_index: int, photo_index: int):
        tuple_index = np.unravel_index(incident_index, self.incident_shape)
        incident_params = {
            axis: self.incident_axes_values[axis][axis_idx]
            for axis, axis_idx in zip(self.incident_axes, tuple_index)
        }
        lnu = self.grid.spectra["incident"][tuple_index]
        delta_log = None
        if self.reference_log10 is not None:
            delta_log = (
                self.grid.log10_specific_ionising_lum["HI"][tuple_index]
                - self.reference_log10
            )
        try:
            photo_params = self.photo_parameter_grid[photo_index]
        except IndexError:
            raise IndexError(
                f"Photoionisation index {photo_index} outside available grid"
            ) from None
        parameters = incident_params | self.fixed_params | photo_params
        return parameters, self.grid.lam, lnu, delta_log


class SobolSampler:
    def __init__(self, metadata: Dict[str, object]):
        self.metadata = metadata
        self.fixed_params = metadata.get("fixed_parameters", {}).copy()
        grid_path = Path(metadata["grid_hdf5"])
        with h5py.File(grid_path, "r") as handle:
            lam = handle["incident_lam"][:]
            lnu = handle["incident_lnu"][:]
            self.lam = lam * Angstrom
            self.lnu = lnu * erg / s / Hz
            parameters_group = handle["parameters"]
            self.parameter_arrays = {
                key: parameters_group[key][:]
                for key in parameters_group.keys()
            }
            if "log10Q_samples" in handle:
                self.log10Q = handle["log10Q_samples"][:]
                self.reference_log10 = handle.attrs.get(
                    "reference_log10_specific_ionising_lum"
                )
            else:
                self.log10Q = None
                self.reference_log10 = None
        self.linelist_path = Path(metadata["linelist_path"])
        self.linelist_name = metadata.get("linelist_name", "linelist.dat")

    def prepare_model(self, incident_index: int, photo_index: int):
        params = {
            key: float(values[incident_index])
            for key, values in self.parameter_arrays.items()
        }
        parameters = params | self.fixed_params
        delta_log = None
        if self.log10Q is not None and self.reference_log10 is not None:
            delta_log = self.log10Q[incident_index] - self.reference_log10
        lnu = self.lnu[incident_index]
        return parameters, self.lam, lnu, delta_log


def _build_index_lists(args, metadata):
    incident_n = metadata["incident_n_models"]
    photo_n = metadata["photoionisation_n_models"]

    if args.list_file is not None:
        loaded = np.loadtxt(args.list_file, dtype=int)
        loaded = np.atleast_2d(loaded)
        if loaded.shape[1] < 2:
            raise ValueError(
                "List file must contain at least two columns (incident, photo)"
            )
        incident_indices = loaded[:, 0]
        photo_indices = loaded[:, 1]
        if args.list_index is not None:
            incident_indices = [incident_indices[args.list_index]]
            photo_indices = [photo_indices[args.list_index]]
        return incident_indices, photo_indices

    if args.incident_index is not None:
        incident_range: Iterable[int] = [args.incident_index]
    else:
        incident_range = range(incident_n)

    if args.photoionisation_index is not None:
        photo_range: Iterable[int] = [args.photoionisation_index]
    else:
        photo_range = range(photo_n)

    incident_indices, photo_indices = np.meshgrid(incident_range, photo_range)
    return incident_indices.flatten(), photo_indices.flatten()


def main():
    parser = argparse.ArgumentParser(
        description="Run Cloudy models on-the-fly"
    )
    parser.add_argument("--cloudy-output-dir", type=str, required=True)
    parser.add_argument("--grid-name", type=str, required=True)
    parser.add_argument("--cloudy-executable-path", type=str, required=True)
    parser.add_argument("--incident-index", type=int, default=None)
    parser.add_argument("--photoionisation-index", type=int, default=None)
    parser.add_argument("--list-file", type=str, default=None)
    parser.add_argument("--list-index", type=int, default=None)
    args = parser.parse_args()

    output_directory = Path(args.cloudy_output_dir) / args.grid_name
    metadata_path = output_directory / "grid_parameters.yaml"
    metadata = _load_metadata(metadata_path)

    cloudy_version = str(metadata["cloudy_version"])
    os.environ["CLOUDY_DATA_PATH"] = (
        f"{args.cloudy_executable_path}/{cloudy_version}/data/:./"
    )
    cloudy_executable = (
        Path(args.cloudy_executable_path)
        / cloudy_version
        / "source"
        / "cloudy.exe"
    )

    sampling_method = metadata.get("sampling_method", "incident")
    if sampling_method == "sobol":
        sampler = SobolSampler(metadata)
    else:
        sampler = IncidentSampler(metadata)

    writer = CloudyInputWriter(output_directory)
    output_dir = output_directory / "output"
    shape_module = _select_cloudy_module(cloudy_version)

    incident_indices, photo_indices = _build_index_lists(args, metadata)

    for incident_index, photo_index in zip(incident_indices, photo_indices):
        incident_index = int(incident_index)
        photo_index = int(photo_index)
        incident_dir = output_directory / str(incident_index)
        incident_dir.mkdir(parents=True, exist_ok=True)

        if sampler.linelist_path is not None:
            linelist_source = sampler.linelist_path
        else:
            linelist_source = Path(
                metadata_path.parent / sampler.linelist_name
            )
        _copy_linelist_if_needed(linelist_source, incident_dir)

        parameters, lam, lnu, delta_log = sampler.prepare_model(
            incident_index, photo_index
        )

        shape_module.ShapeCommands.table_sed(
            "input",
            lam if hasattr(lam, "units") else lam * Angstrom,
            lnu if hasattr(lnu, "units") else lnu * erg / s / Hz,
            output_dir=str(incident_dir),
        )

        writer.write_model(
            incident_index,
            photo_index,
            parameters,
            delta_log,
            cloudy_version,
        )

        print(
            f"Running Cloudy for incident {incident_index}"
            f" / photo {photo_index}"
        )
        result = subprocess.run(
            [str(cloudy_executable), "-r", str(photo_index)],
            cwd=str(incident_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("Cloudy failed:")
            print(result.stdout)
            print(result.stderr)
            continue

        if not _cloudy_run_succeeded(incident_dir, photo_index):
            print(
                f"Cloudy output {incident_dir / f'{photo_index}.out'} "
                "missing 'Cloudy exited OK'; keeping working directory"
            )
            continue

        _store_outputs(
            incident_dir,
            output_dir,
            incident_index,
            photo_index,
            metadata["photoionisation_n_models"],
        )
        try:
            shutil.rmtree(incident_dir)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
