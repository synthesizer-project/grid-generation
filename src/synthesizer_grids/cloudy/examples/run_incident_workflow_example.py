"""Minimal incident-grid workflow example.

This script shows how to wire up the Cloudy workflow helpers so that an
incident spectral grid can be combined with one (or more) Cloudy parameter
files, the metadata emitted, and an optional COSMA7 submission script
generated.

Example usage (adjust the paths for your environment):

```
python -m synthesizer_grids.cloudy.examples.run_incident_workflow_example \
    --incident-grid bc03-2016-Miles_chabrier-0.1,100.hdf5 \
    --incident-grid-dir \
        /cosma7/data/dp004/dc-love2/data/synthesizer_data/grids \
    --cloudy-output-dir /cosma7/data/dp004/dc-love2/data/synth_runs \
    --cloudy-param-file params/c23.01-sps-grid.yaml \
    --cloudy-executable ~/cosma7/codes/c25.00/source/cloudy.exe \
    --machine cosma7 \
    --account dp004 --partition cosma7
```

Running the script will create (or update) a Cloudy output directory named
after `<incident_grid>_cloudy-<param_label>` and, when `--machine` is
provided, write a ready-to-submit SLURM array job that calls `run_cloudy.py`
for each incident index.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synthesizer_grids.cloudy.config import (
    CloudyParameterSet,
    CloudyWorkflowConfig,
    IncidentSamplerConfig,
    SubmissionConfig,
)
from synthesizer_grids.cloudy.workflow import run_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Cloudy metadata and optional SLURM script "
            "from an incident grid"
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
        help="Directory containing incident grids",
    )
    parser.add_argument(
        "--cloudy-output-dir",
        required=True,
        help="Root directory where Cloudy outputs/metadata should be written",
    )
    parser.add_argument(
        "--cloudy-param-file",
        default="params/c23.01-sps-grid.yaml",
        help="Primary Cloudy parameter YAML file",
    )
    parser.add_argument(
        "--cloudy-param-file-extra",
        default=None,
        help="Optional secondary Cloudy parameter YAML override",
    )
    parser.add_argument(
        "--linelist",
        default="create_linelist/linelist-standard.dat",
        help=(
            "Relative or absolute path to the linelist "
            "to copy into each incident directory"
        ),
    )
    parser.add_argument(
        "--cloudy-executable",
        default=None,
        help=(
            "Absolute path to the Cloudy executable "
            "(required when --machine is set)"
        ),
    )
    parser.add_argument(
        "--machine",
        default=None,
        choices=["cosma7", "artemis", None],
        help=(
            "Submission helper to invoke "
            "(omit to skip SLURM script generation)"
        ),
    )
    parser.add_argument(
        "--account",
        default=None,
        help="HPC account (passed to SubmissionConfig)",
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
        help=(
            "Estimated minutes per Cloudy model "
            "(affects COSMA7 walltime estimate)"
        ),
    )
    parser.add_argument(
        "--stride-step",
        type=int,
        default=1000,
        help=(
            "Number of incident models per array stride "
            "when using COSMA7 helpers"
        ),
    )
    parser.add_argument(
        "--disable-striding",
        action="store_true",
        help="Run exactly one incident model per array task (no striding)",
    )
    parser.add_argument(
        "--mail-user",
        default=None,
        help="Optional email address for SLURM notifications",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parameter_set = CloudyParameterSet.from_files(
        args.cloudy_param_file,
        args.cloudy_param_file_extra,
    )

    incident_config = IncidentSamplerConfig(
        incident_grid=args.incident_grid,
        incident_grid_dir=args.incident_grid_dir,
        cloudy_output_dir=args.cloudy_output_dir,
        parameter_files=parameter_set,
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
        mode="incident",
        incident=incident_config,
        submission=submission_config,
    )

    result = run_workflow(workflow_config)

    output_dir = Path(result["output_directory"])
    grid_name = result["new_grid_name"]

    print(f"Prepared Cloudy metadata for {grid_name}")
    print(f"Output directory: {output_dir}")

    array_script = output_dir / "array.slurm"
    if submission_config.machine and array_script.exists():
        print(f"SLURM array script ready: {array_script}")
        print("Submit via: sbatch array.slurm")
    elif submission_config.machine:
        print(
            "Submission helper was requested but array.slurm is missing. "
            "Check the submission configuration."
        )
    else:
        print(
            "Submission helper disabled; run run_cloudy.py manually "
            "or rerun with --machine."
        )


if __name__ == "__main__":
    main()
