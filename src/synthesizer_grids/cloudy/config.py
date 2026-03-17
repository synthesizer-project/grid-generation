"""Configuration helpers and dataclasses for the Cloudy workflow.

These utilities centralise all of the file/path/parameter metadata required by
the various stages (input preparation, execution, and collection) so that the
legacy scripts can share a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

from synthesizer_grids.cloudy.utils import get_cloudy_params

CLOUDY_MODULE_DIR = Path(__file__).resolve().parent


def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates = [Path.cwd() / path, CLOUDY_MODULE_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path


@dataclass
class CloudyParameterSet:
    """Container for the fixed and variable Cloudy parameters."""

    fixed: Dict[str, object]
    variable: Dict[str, Sequence[float]]
    main_file: str
    extra_file: Optional[str] = None

    @classmethod
    def from_files(
        cls,
        main_param_file: str,
        extra_param_file: Optional[str] = None,
    ) -> "CloudyParameterSet":
        """Load and merge one (or two) YAML parameter files."""

        main_path = _resolve_config_path(main_param_file)
        if not main_path.exists():
            raise FileNotFoundError(
                f"Parameter file not found: {main_param_file}"
            )

        fixed, variable = get_cloudy_params(
            main_path.name, param_dir=str(main_path.parent)
        )

        if extra_param_file:
            extra_path = _resolve_config_path(extra_param_file)
            if not extra_path.exists():
                raise FileNotFoundError(
                    f"Extra parameter file not found: {extra_param_file}"
                )
            extra_fixed, extra_variable = get_cloudy_params(
                extra_path.name, param_dir=str(extra_path.parent)
            )
            fixed |= extra_fixed
            variable |= extra_variable

        return cls(
            fixed=fixed,
            variable=variable,
            main_file=main_param_file,
            extra_file=extra_param_file,
        )

    @property
    def label(self) -> str:
        """Return a short identifier for filesystem naming."""

        main = Path(self.main_file).stem
        if self.extra_file:
            extra = Path(self.extra_file).stem
            return f"{main}-{extra}"
        return main


@dataclass
class SubmissionConfig:
    """Configuration for HPC submission helpers."""

    machine: Optional[str] = None
    cloudy_executable_path: Optional[str] = None
    by_photoionisation_grid_point: bool = False
    memory: str = "4G"
    time_per_model: int = 10
    use_striding: bool = True
    stride_step: int = 1000
    account: Optional[str] = None
    partition: Optional[str] = None
    mail_user: Optional[str] = None


@dataclass
class IncidentSamplerConfig:
    """Metadata required to sample from a discrete incident grid."""

    incident_grid: str
    incident_grid_dir: str
    cloudy_output_dir: str
    parameter_files: CloudyParameterSet
    output_linelist: str = "create_linelist/linelist-standard.dat"

    def grid_file(self) -> Path:
        path = Path(self.incident_grid)
        if path.suffix != ".hdf5":
            path = path.with_suffix(".hdf5")
        if not path.is_absolute():
            path = Path(self.incident_grid_dir) / path
        return path

    @property
    def grid_name(self) -> str:
        return Path(self.grid_file()).stem


@dataclass
class SobolSamplerConfig:
    """Metadata required to build Sobol-sampled parameter grids."""

    incident_grid: str
    incident_grid_dir: str
    cloudy_output_dir: str
    cloudy_paramfile: str
    n_samples: int
    seed: Optional[int] = None
    output_linelist: str = "create_linelist/linelist-standard.dat"
    sampling_method: str = "sobol"

    @property
    def param_path(self) -> Path:
        path = _resolve_config_path(self.cloudy_paramfile)
        if path.suffix != ".yaml":
            path = path.with_suffix(".yaml")
        return path

    def output_directory(self, suffix: Optional[str] = None) -> Path:
        stem = Path(self.incident_grid).stem
        param_stem = self.param_path.stem
        parts = [
            stem,
            "cloudy",
            self.sampling_method,
            param_stem,
            f"n{self.n_samples}",
        ]
        if self.seed is not None:
            parts.append(f"seed{self.seed}")
        if suffix:
            parts.append(suffix)
        return Path(self.cloudy_output_dir) / "-".join(parts)


@dataclass
class CloudyWorkflowConfig:
    """High-level configuration passed to the orchestration pipeline."""

    mode: str  # "incident" or "sobol"
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)
    incident: Optional[IncidentSamplerConfig] = None
    sobol: Optional[SobolSamplerConfig] = None

    def validate(self) -> None:
        if self.mode not in {"incident", "sobol"}:
            raise ValueError(f"Unknown workflow mode: {self.mode}")
        if self.mode == "incident" and self.incident is None:
            raise ValueError("Incident mode requires incident config")
        if self.mode == "sobol" and self.sobol is None:
            raise ValueError("Sobol mode requires sobol config")
