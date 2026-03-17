"""Shared helpers for emitting Cloudy input files.

The logic is distilled from the legacy ``create_cloudy_input_grid`` script so
both incident-grid and Sobol workflows consistently generate Abundances,
geometry scaling, YAML sidecars, and the final ``.in`` files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
from synthesizer.abundances import Abundances, depletion_models
from synthesizer.exceptions import InconsistentParameter

from synthesizer_grids.cloudy import cloudy17, cloudy23


class CloudyInputWriter:
    """Thin wrapper around the canonical input-file creation logic."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)

    def _resolve_output_dir(self, incident_index: str | int) -> Path:
        return self.output_root / str(incident_index)

    def write_model(
        self,
        incident_index: int,
        photoionisation_index: int,
        parameters: Dict[str, object],
        delta_log10_specific_ionising_luminosity: Optional[float],
        cloudy_version: str,
    ) -> Path:
        """Persist YAML + Cloudy input for a single model."""

        target_dir = self._resolve_output_dir(incident_index)
        target_dir.mkdir(parents=True, exist_ok=True)

        parameters = self._prepare_abundance_scalings(parameters)

        if parameters.get("output_linelist"):
            linelist_name = Path(parameters["output_linelist"]).name
            parameters["output_linelist"] = str(target_dir / linelist_name)

        abundances = Abundances(
            metallicity=float(parameters["metallicities"]),
            reference=parameters["reference_abundance"],
            alpha=parameters["alpha_enhancement"],
            abundances=(
                parameters["abundance_scalings"]
                if parameters["abundance_scalings"]
                else None
            ),
            depletion_model=depletion_models.Gutkin2016(),
        )

        parameters["ionisation_parameter"] = (
            self._calculate_ionisation_parameter(
                parameters, delta_log10_specific_ionising_luminosity
            )
        )

        yaml_path = target_dir / f"{photoionisation_index}.yaml"
        self._write_parameters_yaml(yaml_path, parameters)

        if cloudy_version == "c17.03":
            cloudy_module = cloudy17
        elif cloudy_version in {
            "c23.01",
            "c25.00",
        } or cloudy_version.startswith("c23"):
            cloudy_module = cloudy23
        else:
            raise InconsistentParameter(
                f"Unsupported cloudy version '{cloudy_version}' "
                "for input generation"
            )

        shape_commands = ['table SED "input.sed" \n']

        cloudy_module.create_cloudy_input(
            str(photoionisation_index),
            shape_commands,
            abundances,
            output_dir=str(target_dir),
            **parameters,
        )

        return target_dir

    @staticmethod
    def _prepare_abundance_scalings(
        parameters: Dict[str, object],
    ) -> Dict[str, object]:
        parameters = parameters.copy()
        parameters.setdefault("abundance_scalings", {})
        for key, value in list(parameters.items()):
            parts = key.split(".")
            if len(parts) == 2 and parts[0] == "abundance_scalings":
                parameters["abundance_scalings"][parts[1]] = value
        return parameters

    @staticmethod
    def _calculate_ionisation_parameter(
        parameters, delta_log10_specific_ionising_luminosity
    ):
        model = parameters["ionisation_parameter_model"]
        geometry = parameters["geometry"]

        if model == "ref":
            reference = parameters["reference_ionisation_parameter"]
            if geometry in {"spherical", "spherical-U"}:
                power = 1 / 3
            elif geometry == "planeparallel":
                power = 1
            else:
                raise InconsistentParameter(
                    f"Unknown geometry choice: {geometry}"
                )
            return 10 ** (
                np.log10(reference)
                + power * delta_log10_specific_ionising_luminosity
            )
        if model == "fixed":
            return parameters["ionisation_parameter"]
        raise InconsistentParameter(
            "ERROR: do not understand U model choice: "
            f"{parameters['ionisation_parameter_model']}"
        )

    @staticmethod
    def _write_parameters_yaml(
        path: Path, parameters: Dict[str, object]
    ) -> None:
        serialisable = {}
        for key, value in parameters.items():
            if isinstance(value, np.floating):
                serialisable[key] = float(value)
            else:
                serialisable[key] = value
        with open(path, "w") as handle:
            yaml.dump(serialisable, handle, default_flow_style=False)
