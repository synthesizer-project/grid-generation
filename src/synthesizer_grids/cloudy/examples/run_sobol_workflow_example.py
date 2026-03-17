"""Sobol-sampled Cloudy workflow example.

This script demonstrates how to prepare a Sobol parameter sweep that draws
continuous samples from the ranges defined in a YAML file, interpolates the
incident spectra, and emits both the metadata (`grid_parameters.yaml`) and the
HDF5 parameter grid consumed by `run_cloudy.py`.

Example usage:

```
python -m synthesizer_grids.cloudy.examples.run_sobol_workflow_example \
    --incident-grid bc03-2016-Miles_chabrier-0.1,100.hdf5 \
    --incident-grid-dir \
        /cosma7/data/dp004/dc-love2/data/synthesizer_data/grids \
    --cloudy-output-dir /cosma7/data/dp004/dc-love2/data/synth_runs \
    --sobol-param-file params/c25.00-sps-sobol.yaml \
    --n-samples 2048 --seed 42 \
    --cloudy-executable ~/cosma7/codes/c25.00/source/cloudy.exe \
    --machine cosma7 --account dp004 --partition cosma7
```

When `--machine` is provided a SLURM array script (`array.slurm`) is written in
`<cloudy_output_dir>/<derived_grid_name>/` ready for submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synthesizer_grids.cloudy.config import (
    CloudyWorkflowConfig,
    SobolSamplerConfig,
    SubmissionConfig,
)
from synthesizer_grids.cloudy.workflow import run_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Sobol metadata/HDF5 and an optional "
            "COSMA7 submission script"
        ),
    )
    parser.add_argument(
        "--incident-grid",
        required=True,
        help="Incident grid filename (with or without .hdf5 suffix)",
    )
    parser.add_argument(
        "--incident-grid-dir",
        required=True,
        help="Directory that stores the incident grid",
    )
    parser.add_argument(
        "--cloudy-output-dir",
        required=True,
        help="Root directory where Sobol workflow outputs should be written",
    )
    parser.add_argument(
        "--sobol-param-file",
        default="params/c25.00-sps-sobol.yaml",
        help="YAML file containing Sobol sampling ranges and fixed parameters",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Number of Sobol samples to draw",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional Sobol scrambling seed",
    )
    parser.add_argument(
        "--linelist",
        default="create_linelist/linelist-standard.dat",
        help="Path to the linelist to copy into each incident directory",
    )
    parser.add_argument(
        "--cloudy-executable",
        default=None,
        help="Absolute path to cloudy.exe (required when --machine is set)",
    )
    parser.add_argument(
        "--machine",
        default=None,
        choices=["cosma7", "artemis", None],
        help="Submission helper to invoke (omit to skip script generation)",
    )
    parser.add_argument(
        "--account",
        default=None,
        help="HPC account/project string",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="HPC partition/queue name",
    )
    parser.add_argument(
        "--time-per-model",
        type=int,
        default=10,
        help="Estimated minutes per Cloudy model for walltime budgeting",
    )
    parser.add_argument(
        "--stride-step",
        type=int,
        default=1000,
        help="Incident models per task when striding is enabled",
    )
    parser.add_argument(
        "--disable-striding",
        action="store_true",
        help="Run exactly one incident index per array task",
    )
    parser.add_argument(
        "--mail-user",
        default=None,
        help="Email address for SLURM notifications",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sobol_config = SobolSamplerConfig(
        incident_grid=args.incident_grid,
        incident_grid_dir=args.incident_grid_dir,
        cloudy_output_dir=args.cloudy_output_dir,
        cloudy_paramfile=args.sobol_param_file,
        n_samples=args.n_samples,
        seed=args.seed,
        output_linelist=args.linelist,
    )

    submission_config = SubmissionConfig(
        machine=args.machine,
        cloudy_executable_path=args.cloudy_executable,
        account=args.account,
        partition=args.partition,
        time_per_model=args.time_per_model,
        use_striding=not args.disable_striding,
        stride_step=args.stride_step,
        mail_user=args.mail_user,
    )

    workflow_config = CloudyWorkflowConfig(
        mode="sobol",
        sobol=sobol_config,
        submission=submission_config,
    )

    result = run_workflow(workflow_config)

    output_dir = Path(result["output_directory"])
    grid_name = result["new_grid_name"]

    print(f"Prepared Sobol Cloudy metadata for {grid_name}")
    print(f"Output directory: {output_dir}")
    print(f"Parameter HDF5: {result['grid_hdf5']}")

    array_script = output_dir / "array.slurm"
    if submission_config.machine and array_script.exists():
        print(f"SLURM array script ready: {array_script}")
        print("Submit via: sbatch array.slurm")
    elif submission_config.machine:
        print(
            "Submission helper was requested but array.slurm is missing. "
            "Verify COSMA7 submission configuration."
        )
    else:
        print(
            "Submission helper disabled; run run_cloudy.py manually "
            "or rerun with --machine."
        )


if __name__ == "__main__":
    main()
