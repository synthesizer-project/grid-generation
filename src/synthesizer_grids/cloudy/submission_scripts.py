"""Functions for creating submission scripts for specific machines."""

from math import ceil
from pathlib import Path

RUN_CLOUDY_PATH = Path(__file__).resolve().parent / "run_cloudy.py"


def _build_run_cloudy_slurm_script(
    *,
    new_grid_name: str,
    total_incident_points: int,
    photoionisation_per_incident: int,
    cloudy_output_dir: str,
    cloudy_executable_path: str,
    partition: str,
    account: str | None,
    memory: str,
    time_per_model: int,
    use_striding: bool,
    stride_step: int,
    mail_user: str | None,
    python_env: str | None = None,
) -> str:
    """Create a SLURM array script that loops incident (and photo) indices."""

    if total_incident_points <= 0:
        raise ValueError("total_incident_points must be positive")
    if photoionisation_per_incident <= 0:
        raise ValueError("photoionisation_per_incident must be positive")
    if stride_step <= 0:
        raise ValueError("stride_step must be positive")

    total_models = total_incident_points * photoionisation_per_incident
    if use_striding and stride_step < total_incident_points:
        n_array_jobs = max(1, stride_step)
    else:
        n_array_jobs = total_incident_points

    models_per_job = ceil(total_models / n_array_jobs)
    total_minutes = int(models_per_job * time_per_model * 1.2)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    walltime = f"{hours:02d}:{minutes:02d}:00"

    array_range = f"0-{n_array_jobs - 1}"
    grid_dir = Path(cloudy_output_dir) / new_grid_name
    run_cloudy_cmd = (
        f'python "{RUN_CLOUDY_PATH}" '
        f'--grid-name "{new_grid_name}" '
        f'--cloudy-output-dir "{cloudy_output_dir}" '
        f'--cloudy-executable-path "{cloudy_executable_path}" '
    )

    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={new_grid_name[:16]}",
        f"#SBATCH --array={array_range}",
        f"#SBATCH -p {partition}",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH -t {walltime}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --output={grid_dir}/logs/log%A_%a.out",
        f"#SBATCH --error={grid_dir}/logs/log%A_%a.err",
    ]

    if account:
        script_lines.insert(4, f"#SBATCH -A {account}")

    if mail_user:
        script_lines.extend(
            [
                "#SBATCH --mail-type=ALL",
                f"#SBATCH --mail-user={mail_user}",
            ]
        )

    script_lines.extend(
        [
            "",
            f'GRID_DIR="{grid_dir}"',
            'mkdir -p "$GRID_DIR/logs"',
            'cd "$GRID_DIR"',
        ]
    )

    if python_env:
        script_lines.extend([f"source {python_env}/bin/activate", ""])

    script_lines.extend(
        [
            f"TOTAL_MODELS={total_models}",
            f"PHOTOIONISATION_PER_INCIDENT={photoionisation_per_incident}",
            f"ARRAY_SIZE={n_array_jobs}",
            "task_id=${SLURM_ARRAY_TASK_ID}",
            "BASE_CHUNK=$((TOTAL_MODELS / ARRAY_SIZE))",
            "REMAINDER=$((TOTAL_MODELS % ARRAY_SIZE))",
            "if [ $task_id -lt $REMAINDER ]; then",
            "    START=$((task_id * (BASE_CHUNK + 1)))",
            "    STOP=$((START + BASE_CHUNK + 1))",
            "else",
            "    START=$((REMAINDER * (BASE_CHUNK + 1)"
            " + (task_id - REMAINDER) * BASE_CHUNK))",
            "    STOP=$((START + BASE_CHUNK))",
            "fi",
            "if [ $START -ge $TOTAL_MODELS ]; then",
            '    echo "No models assigned to task $task_id"',
            "    exit 0",
            "fi",
            "if [ $STOP -gt $TOTAL_MODELS ]; then",
            "    STOP=$TOTAL_MODELS",
            "fi",
            "for ((model_idx=START; model_idx<STOP; model_idx++)); do",
            "    incident_idx=$((model_idx / PHOTOIONISATION_PER_INCIDENT))",
            "    photo_idx=$((model_idx % PHOTOIONISATION_PER_INCIDENT))",
        ]
    )
    script_lines.extend(
        [
            f"    {run_cloudy_cmd}--incident-index $incident_idx"
            " --photoionisation-index $photo_idx || exit 1",
            "done",
            "",
        ]
    )

    script_content = "\n".join(script_lines)
    script_path = grid_dir / "array.slurm"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as handle:
        handle.write(script_content)

    print(f"Created SLURM job script: {script_path}")
    return str(script_path)


def artemis(
    new_grid_name,
    number_of_incident_grid_points,
    number_of_photoionisation_models,
    cloudy_output_dir,
    cloudy_executable_path,
    memory="4G",
    by_photoionisation_grid_point=False,
    partition=None,
    account=None,
    time_per_model=10,
    use_striding=True,
    stride_step=1000,
    mail_user=None,
):
    """Submission script generator for the Artemis cluster."""

    if by_photoionisation_grid_point:
        raise NotImplementedError(
            "Per-photoionisation array jobs are no longer supported"
        )

    models_per_incident = max(1, int(number_of_photoionisation_models))
    total_incident_points = int(number_of_incident_grid_points)

    if partition is None:
        if models_per_incident < 5:
            partition = "short"
        elif models_per_incident < 33:
            partition = "general"
        else:
            partition = "long"

    return _build_run_cloudy_slurm_script(
        new_grid_name=new_grid_name,
        total_incident_points=total_incident_points,
        photoionisation_per_incident=models_per_incident,
        cloudy_output_dir=cloudy_output_dir,
        cloudy_executable_path=cloudy_executable_path,
        partition=partition,
        account=account,
        memory=memory,
        time_per_model=time_per_model,
        use_striding=use_striding,
        stride_step=stride_step,
        mail_user=mail_user,
    )


def cosma7(
    new_grid_name,
    number_of_incident_grid_points,
    number_of_photoionisation_models,
    cloudy_output_dir,
    cloudy_executable_path,
    memory="4G",
    by_photoionisation_grid_point=False,
    partition="cosma7",
    account="dp004",
    time_per_model=10,
    use_striding=True,
    stride_step=1000,
    mail_user=None,
    python_env=None,
):
    """Generate a COSMA7 submission script that runs Cloudy on-the-fly."""

    if by_photoionisation_grid_point:
        raise NotImplementedError(
            "Per-photoionisation array jobs are no longer supported"
        )

    models_per_incident = max(1, int(number_of_photoionisation_models))
    total_incident_points = int(number_of_incident_grid_points)

    return _build_run_cloudy_slurm_script(
        new_grid_name=new_grid_name,
        total_incident_points=total_incident_points,
        photoionisation_per_incident=models_per_incident,
        cloudy_output_dir=cloudy_output_dir,
        cloudy_executable_path=cloudy_executable_path,
        partition=partition,
        account=account,
        memory=memory,
        time_per_model=time_per_model,
        use_striding=use_striding,
        stride_step=stride_step,
        mail_user=mail_user,
        python_env=python_env,
    )
