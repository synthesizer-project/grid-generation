"""High-level orchestration helpers for Cloudy workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml
from synthesizer.emission_models import IncidentEmission
from synthesizer.emissions import Sed
from synthesizer.grid import Grid
from synthesizer.particle import Stars
from unyt import Angstrom, Hz, Msun, erg, eV, s, yr

from synthesizer_grids.cloudy import submission_scripts
from synthesizer_grids.cloudy.config import (
    CloudyWorkflowConfig,
    IncidentSamplerConfig,
    SobolSamplerConfig,
    SubmissionConfig,
)
from synthesizer_grids.cloudy.utils import get_grid_props_cloudy


def _serialise_value(value):
    if isinstance(value, dict):
        return {key: _serialise_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise_value(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _serialise_dict(data: Dict[str, object]) -> Dict[str, object]:
    return {key: _serialise_value(val) for key, val in data.items()}


def _resolve_linelist_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    return path


def run_incident_workflow(
    config: IncidentSamplerConfig,
    submission: SubmissionConfig,
) -> Dict[str, object]:
    grid = Grid(
        config.grid_name,
        grid_dir=config.incident_grid_dir,
        ignore_lines=True,
    )

    incident_axes = grid.axes
    incident_axes_values = grid.axes_values
    (
        _,
        _,
        incident_n_models,
        _,
        _,
        _,
    ) = get_grid_props_cloudy(
        incident_axes, incident_axes_values, verbose=False
    )

    fixed_params = config.parameter_files.fixed.copy()
    variable_params = config.parameter_files.variable.copy()

    if variable_params:
        photo_axes = list(variable_params.keys())
        (
            _,
            _,
            photoionisation_n_models,
            _,
            _,
            _,
        ) = get_grid_props_cloudy(photo_axes, variable_params, verbose=False)
    else:
        photo_axes = []
        photoionisation_n_models = 1

    new_grid_name = f"{config.grid_name}_cloudy-{config.parameter_files.label}"
    output_directory = Path(config.cloudy_output_dir) / new_grid_name
    output_directory.mkdir(parents=True, exist_ok=True)

    linelist_src = _resolve_linelist_path(config.output_linelist)
    metadata = _serialise_dict(
        fixed_params | variable_params | incident_axes_values
    )
    metadata |= {
        "incident_n_models": int(incident_n_models),
        "photoionisation_n_models": int(photoionisation_n_models),
        "sampling_method": "incident",
        "incident_axes": list(incident_axes),
        "incident_axes_values": {
            axis: _serialise_value(incident_axes_values[axis])
            for axis in incident_axes
        },
        "incident_grid_path": str(config.grid_file().resolve()),
        "linelist_path": str(linelist_src.resolve()),
        "linelist_name": linelist_src.name,
        "parameter_files": {
            "main": config.parameter_files.main_file,
            "extra": config.parameter_files.extra_file,
        },
        "fixed_parameters": _serialise_dict(fixed_params),
        "variable_parameters": _serialise_dict(variable_params),
    }
    with open(output_directory / "grid_parameters.yaml", "w") as handle:
        yaml.dump(metadata, handle, default_flow_style=False)

    if submission.machine:
        getattr(submission_scripts, submission.machine)(
            new_grid_name=new_grid_name,
            number_of_incident_grid_points=int(incident_n_models),
            number_of_photoionisation_models=int(photoionisation_n_models),
            cloudy_output_dir=config.cloudy_output_dir,
            cloudy_executable_path=submission.cloudy_executable_path,
            memory=submission.memory,
            by_photoionisation_grid_point=submission.by_photoionisation_grid_point,
            partition=submission.partition,
            account=submission.account,
            time_per_model=submission.time_per_model,
            use_striding=submission.use_striding,
            stride_step=submission.stride_step,
            mail_user=submission.mail_user,
        )

    return {
        "new_grid_name": new_grid_name,
        "output_directory": str(output_directory),
    }


def _parse_sobol_ranges(params, incident_grid: Grid):
    incident_params = {}
    photoionisation_params = {}
    fixed_params = {}

    incident_axes = set(incident_grid.axes)
    for axis in incident_grid.axes:
        axis_values = getattr(incident_grid, axis)
        min_val = float(np.min(axis_values))
        max_val = float(np.max(axis_values))
        scale = (
            "log"
            if axis in ["ages", "age", "metallicities", "metallicity"]
            else "linear"
        )
        incident_params[axis] = (min_val, max_val, scale)

    for key, value in params.items():
        is_incident = False
        incident_key = None
        if key in ["age", "ages"] and (
            "ages" in incident_axes or "age" in incident_axes
        ):
            is_incident = True
            incident_key = "ages" if "ages" in incident_axes else "age"
        elif key in ["metallicity", "metallicities"] and (
            "metallicities" in incident_axes or "metallicity" in incident_axes
        ):
            is_incident = True
            incident_key = (
                "metallicities"
                if "metallicities" in incident_axes
                else "metallicity"
            )

        if isinstance(value, list) and len(value) == 3:
            min_val, max_val, scale = value
            if is_incident:
                incident_params[incident_key] = (min_val, max_val, scale)
            else:
                photoionisation_params[key] = (min_val, max_val, scale)
        elif key == "abundance_scalings" and isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, list) and len(sub_val) == 3:
                    min_val, max_val, scale = sub_val
                    photoionisation_params[f"abundance_scalings.{sub_key}"] = (
                        min_val,
                        max_val,
                        scale,
                    )
                else:
                    fixed_params.setdefault("abundance_scalings", {})[
                        sub_key
                    ] = sub_val
        else:
            fixed_params[key] = value

    return incident_params, photoionisation_params, fixed_params


def _generate_sobol_samples(
    incident_params: Dict[str, Tuple[float, float, str]],
    photo_params: Dict[str, Tuple[float, float, str]],
    n_samples: int,
    seed: Optional[int],
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    from scipy.stats import qmc

    all_params = incident_params | photo_params
    sampler = qmc.Sobol(d=len(all_params), scramble=True, seed=seed)
    n_sobol = int(2 ** np.ceil(np.log2(n_samples)))
    unit = sampler.random(n=n_sobol)[:n_samples]

    names = list(all_params.keys())
    samples: List[Dict[str, float]] = []
    for idx in range(n_samples):
        sample: Dict[str, float] = {}
        for dim, name in enumerate(names):
            min_val, max_val, scale = all_params[name]
            if scale == "log":
                log_min = np.log10(min_val)
                log_max = np.log10(max_val)
                value = 10 ** (log_min + unit[idx, dim] * (log_max - log_min))
            else:
                value = min_val + unit[idx, dim] * (max_val - min_val)
            sample[name] = value
        samples.append(sample)

    incident_names = set(incident_params.keys())
    incident_samples = [
        {k: v for k, v in sample.items() if k in incident_names}
        for sample in samples
    ]
    photo_samples = [
        {k: v for k, v in sample.items() if k not in incident_names}
        for sample in samples
    ]
    return incident_samples, photo_samples


def _interpolate_sobol_spectra(
    incident_grid: Grid,
    incident_samples: List[Dict[str, float]],
    fixed_params: Dict[str, object],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float]]:
    initial_masses = np.ones(len(incident_samples)) * Msun
    ages_array = np.array(
        [
            sample.get("ages") or sample.get("age")
            for sample in incident_samples
        ]
    )
    metallicities_array = np.array(
        [
            sample.get("metallicities") or sample.get("metallicity")
            for sample in incident_samples
        ]
    )

    stars = Stars(
        initial_masses=initial_masses,
        ages=ages_array * yr,
        metallicities=metallicities_array,
    )
    emission_model = IncidentEmission(incident_grid, per_particle=True)
    spectra = stars.get_spectra(emission_model, nthreads=-1)
    incident_lnu = spectra.lnu.value

    if fixed_params.get("ionisation_parameter_model") == "ref":
        ref_age = fixed_params.get("reference_age")
        ref_met = fixed_params.get("reference_metallicity")
        if ref_age is None or ref_met is None:
            raise ValueError(
                "reference_age and reference_metallicity required "
                "for 'ref' model"
            )
        ref_age = float(ref_age)
        ref_met = float(ref_met)
        ref_dict = {}
        for axis in incident_grid.axes:
            if axis in ["ages", "age"]:
                ref_dict[axis] = ref_age
            elif axis in ["metallicities", "metallicity"]:
                ref_dict[axis] = ref_met
        ref_point = incident_grid.get_grid_point(**ref_dict)
        reference_log10Q = incident_grid.log10_specific_ionising_lum["HI"][
            ref_point
        ]

        lam_units = incident_grid.lam
        if not hasattr(lam_units, "units"):
            lam_units = lam_units * Angstrom
        log10Q_samples = np.zeros(len(incident_samples))
        for ii in range(len(incident_samples)):
            sed = Sed(
                lam=lam_units,
                lnu=incident_lnu[ii] * erg / s / Hz,
            )
            q_hi = sed.calculate_ionising_photon_production_rate(
                ionisation_energy=13.6 * eV,
                limit=100,
            )
            log10Q_samples[ii] = np.log10(q_hi)
    else:
        reference_log10Q = None
        log10Q_samples = None

    return incident_lnu, log10Q_samples, reference_log10Q


def run_sobol_workflow(
    config: SobolSamplerConfig,
    submission: SubmissionConfig,
) -> Dict[str, object]:
    incident_grid = Grid(
        config.incident_grid,
        grid_dir=config.incident_grid_dir,
        ignore_lines=True,
    )
    with open(config.param_path, "r") as handle:
        sobol_params = yaml.safe_load(handle)

    incident_params, photo_params, fixed_params = _parse_sobol_ranges(
        sobol_params, incident_grid
    )
    incident_samples, photo_samples = _generate_sobol_samples(
        incident_params, photo_params, config.n_samples, config.seed
    )

    lam = incident_grid.lam
    incident_lnu, log10Q_samples, reference_log10Q = (
        _interpolate_sobol_spectra(
            incident_grid, incident_samples, fixed_params
        )
    )

    new_grid_name = (
        f"{Path(config.incident_grid).stem}_cloudy-sobol-"
        f"{config.param_path.stem}-n{config.n_samples}"
    )
    if config.seed is not None:
        new_grid_name += f"-seed{config.seed}"
    output_directory = Path(config.cloudy_output_dir) / new_grid_name
    output_directory.mkdir(parents=True, exist_ok=True)

    linelist_src = _resolve_linelist_path(config.output_linelist)
    grid_hdf5_path = output_directory / f"{new_grid_name}.hdf5"
    metadata = {
        "incident_n_models": int(config.n_samples),
        "photoionisation_n_models": 1,
        "total_n_models": int(config.n_samples),
        "sampling_method": "sobol",
        "seed": config.seed,
        **_serialise_dict(fixed_params),
        "incident_params": _serialise_dict(incident_params),
        "photoionisation_params": _serialise_dict(photo_params),
        "depletion_model": fixed_params.get("depletion_model"),
        "grid_hdf5": str(grid_hdf5_path),
        "linelist_path": str(linelist_src.resolve()),
        "linelist_name": linelist_src.name,
        "fixed_parameters": _serialise_dict(fixed_params),
    }
    with open(output_directory / "grid_parameters.yaml", "w") as handle:
        yaml.dump(metadata, handle, default_flow_style=False)

    with h5py.File(grid_hdf5_path, "w") as hf:
        param_group = hf.create_group("parameters")
        for i, (inc_sample, photo_sample) in enumerate(
            zip(incident_samples, photo_samples)
        ):
            combined = {**inc_sample, **photo_sample}
            for key, value in combined.items():
                if key not in param_group:
                    param_group.create_dataset(
                        key,
                        data=np.zeros(config.n_samples),
                        compression="gzip",
                    )
                param_group[key][i] = float(value)

        for key, value in fixed_params.items():
            if value is None:
                continue
            if isinstance(value, dict):
                hf.attrs[key] = json.dumps(value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                hf.attrs[key] = np.array(value)
            else:
                hf.attrs[key] = value

        hf.attrs["n_samples"] = int(config.n_samples)
        hf.attrs["sampling_method"] = "sobol"
        if config.seed is not None:
            hf.attrs["seed"] = config.seed
        if reference_log10Q is not None:
            hf.attrs["reference_log10_specific_ionising_lum"] = (
                reference_log10Q
            )
        if log10Q_samples is not None:
            hf.create_dataset(
                "log10Q_samples",
                data=log10Q_samples,
                compression="gzip",
            )
        hf.create_dataset("incident_lam", data=lam, compression="gzip")
        hf.create_dataset(
            "incident_lnu", data=incident_lnu, compression="gzip"
        )

    if submission.machine:
        getattr(submission_scripts, submission.machine)(
            new_grid_name=new_grid_name,
            number_of_incident_grid_points=int(config.n_samples),
            number_of_photoionisation_models=1,
            cloudy_output_dir=config.cloudy_output_dir,
            cloudy_executable_path=submission.cloudy_executable_path,
            memory=submission.memory,
            by_photoionisation_grid_point=False,
            partition=submission.partition,
            account=submission.account,
            time_per_model=submission.time_per_model,
            use_striding=submission.use_striding,
            stride_step=submission.stride_step,
            mail_user=submission.mail_user,
        )

    return {
        "new_grid_name": new_grid_name,
        "output_directory": str(output_directory),
        "grid_hdf5": str(grid_hdf5_path),
    }


def run_workflow(config: CloudyWorkflowConfig) -> Dict[str, object]:
    config.validate()
    if config.mode == "incident":
        assert config.incident is not None
        return run_incident_workflow(config.incident, config.submission)
    assert config.sobol is not None
    return run_sobol_workflow(config.sobol, config.submission)
