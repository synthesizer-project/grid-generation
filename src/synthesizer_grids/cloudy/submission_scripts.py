"""
Functions for creating submission scripts for specific machines.
"""


def create_slurm_job_script_by_incident_grid_point(
    number_of_incident_grid_points=None,
    partition=None,
    new_grid_name=None,
    cloudy_output_dir=None,
    cloudy_executable_path=None,
    memory="4G",
):
    """
    Create a generic slurm input script where we loop over every
    photoionisation model for a given incident grid point. In this case the
    number of jobs is the number of incident grid points and the number of
    cloudy runs is the number of photoionisation grid points.
    """

    number_of_jobs = number_of_incident_grid_points

    slurm_job_script = "\n".join(
        [
            "#!/bin/bash",
            "#SBATCH --job-name=run_cloudy      # Job name",
            "#SBATCH --output=output/%A_%a.out  # Standard output log",
            "#SBATCH --error=output/%A_%a.err   # Error log",
            f"#SBATCH --array=0-{number_of_jobs-1}  # Job array range",
            "#SBATCH --ntasks=1                 # Number of tasks per job",
            "#SBATCH --cpus-per-task=1          # CPU cores per task",
            f"#SBATCH --mem={memory}             # Memory per task",
            f"#SBATCH --partition={partition}    # Partition/queue name",
            "",
            "# Run command",
            "python run_cloudy.py \\",
            f"    --grid-name={new_grid_name} \\",
            f"    --cloudy-output-dir={cloudy_output_dir} \\",
            f"    --cloudy-executable-path={cloudy_executable_path} \\",
            "    --incident-index=${SLURM_ARRAY_TASK_ID}",
        ]
    )
    return slurm_job_script


def create_slurm_job_script_by_photoionisation_grid_point(
    number_of_photoionisation_models=None,
    partition=None,
    new_grid_name=None,
    cloudy_output_dir=None,
    cloudy_executable_path=None,
    memory="4G",
):
    """
    Create a generic slurm input script where we loop over every
    photoionisation model for a given photoionisation grid point. In this
    case the number of jobs is the number of photoionisation grid points and
    the number of cloudy runs per jobs is the number of incident grid points.
    """

    number_of_jobs = number_of_photoionisation_models

    slurm_job_script = "\n".join(
        [
            "#!/bin/bash",
            "#SBATCH --job-name=run_cloudy      # Job name",
            "#SBATCH --output=output/%A_%a.out  # Standard output log",
            "#SBATCH --error=output/%A_%a.err   # Error log",
            f"#SBATCH --array=0-{number_of_jobs-1} # Job array range",
            "#SBATCH --ntasks=1                 # Number of tasks per job",
            "#SBATCH --cpus-per-task=1          # CPU cores per task",
            f"#SBATCH --mem={memory}             # Memory per task",
            f"#SBATCH --partition={partition}    # Partition/queue name",
            "",
            "# Run command",
            "python run_cloudy.py \\",
            f"    --grid-name={new_grid_name} \\",
            f"    --cloudy-output-dir={cloudy_output_dir} \\",
            f"    --cloudy-executable-path={cloudy_executable_path} \\",
            "    --photoionisation-index=${SLURM_ARRAY_TASK_ID}",
        ]
    )
    return slurm_job_script


def create_slurm_job_script_for_list(
    list_file=None,
    number_of_jobs=None,
    partition=None,
    new_grid_name=None,
    cloudy_output_dir=None,
    cloudy_executable_path=None,
    memory="4G",
):
    """
    Create a generic slurm input script where we use a list of models.
    """

    slurm_job_script = "\n".join(
        [
            "#!/bin/bash",
            "#SBATCH --job-name=run_cloudy      # Job name",
            "#SBATCH --output=output/%A_%a.out  # Standard output log",
            "#SBATCH --error=output/%A_%a.err   # Error log",
            f"#SBATCH --array=0-{number_of_jobs-1}  # Job array range",
            "#SBATCH --ntasks=1                 # Number of tasks per job",
            "#SBATCH --cpus-per-task=1          # CPU cores per task",
            f"#SBATCH --mem={memory}             # Memory per task",
            f"#SBATCH --partition={partition}    # Partition/queue name",
            "",
            "# Run command",
            "python run_cloudy.py \\",
            f"    --grid-name={new_grid_name} \\",
            f"    --cloudy-output-dir={cloudy_output_dir} \\",
            f"    --cloudy-executable-path={cloudy_executable_path} \\",
            f"    --list-file={list_file} \\",
            "    --list-index=${SLURM_ARRAY_TASK_ID}",
        ]
    )
    return slurm_job_script


def artemis(
    new_grid_name,
    number_of_jobs=None,
    number_of_incident_grid_points=None,
    number_of_photoionisation_models=None,
    cloudy_output_dir=None,
    cloudy_executable_path=None,
    memory="4G",
    by_photoionisation_grid_point=False,
    from_list=None,
):
    """
    Submission script generator for artemis
    """

    # If a number_of_jobs is provided use this and assume the number of models
    # is 1. This is mostly relevant for re-running certain jobs due to cloudy
    # failures.
    if number_of_jobs is not None:
        number_of_models = 1
    else:
        if by_photoionisation_grid_point:
            number_of_models = number_of_incident_grid_points
            number_of_jobs = number_of_photoionisation_models
        else:
            number_of_models = number_of_photoionisation_models
            number_of_jobs = number_of_incident_grid_points

    # determine the partition to use:
    # short = 2 hours
    if number_of_models < 5:
        partition = "short"

    # general = 8 hours
    elif number_of_models < 33:
        partition = "general"

    # long = 8 days
    else:
        partition = "long"

    # create job script
    if from_list is not None:
        slurm_job_script = create_slurm_job_script_for_list(
            list_file=from_list,
            number_of_jobs=number_of_jobs,
            partition=partition,
            new_grid_name=new_grid_name,
            cloudy_output_dir=cloudy_output_dir,
            cloudy_executable_path=cloudy_executable_path,
            memory=memory,
        )

    else:
        if by_photoionisation_grid_point:
            slurm_job_script = (
                create_slurm_job_script_by_photoionisation_grid_point(
                    number_of_jobs,
                    partition=partition,
                    new_grid_name=new_grid_name,
                    cloudy_output_dir=cloudy_output_dir,
                    cloudy_executable_path=cloudy_executable_path,
                    memory=memory,
                )
            )
        else:
            slurm_job_script = create_slurm_job_script_by_incident_grid_point(
                number_of_jobs,
                partition=partition,
                new_grid_name=new_grid_name,
                cloudy_output_dir=cloudy_output_dir,
                cloudy_executable_path=cloudy_executable_path,
                memory=memory,
            )

    # save job script
    open(f"{new_grid_name}.slurm", "w").write(slurm_job_script)


def cosma7_sobol(
    new_grid_name,
    n_samples,
    cloudy_output_dir,
    cloudy_executable_path,
    account="dp276",
    partition="cosma7",
    time_per_model=10,
    use_striding=True,
    stride_step=1000,
    mail_user=None,
):
    """
    Create a COSMA7 SLURM job script for running Sobol-sampled Cloudy models.

    Args:
        new_grid_name (str): Name of the grid
        n_samples (int): Total number of Sobol samples
        cloudy_output_dir (str): Base directory containing grid outputs
        cloudy_executable_path (str): Path to cloudy executable directory
        account (str): SLURM account (default: dp004)
        partition (str): SLURM partition (default: cosma7)
        time_per_model (int): Time per model in minutes (default: 10)
        use_striding (bool): Use strided array jobs (default: True)
        stride_step (int): Stride step for array jobs (default: 1000)
        mail_user (str): Email for notifications (optional)
    """

    grid_dir = f"{cloudy_output_dir}/{new_grid_name}"

    # Determine if striding is needed based on n_samples vs stride_step
    use_striding_actual = use_striding and n_samples > stride_step

    # Determine array job configuration and models per job
    if use_striding_actual:
        n_array_jobs = min(stride_step, n_samples)
        array_range = f"0-{n_array_jobs - 1}"
        # Each array job handles every stride_step-th model
        models_per_job = (n_samples + stride_step - 1) // stride_step  # Ceiling division
    else:
        array_range = f"0-{n_samples - 1}"
        models_per_job = 1

    # Calculate walltime based on models per job
    # Add 20% buffer for overhead
    total_minutes = int(models_per_job * time_per_model * 1.2)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    walltime = f"{hours:02d}:{minutes:02d}:00"

    print(f"Models per array job: {models_per_job}")
    print(f"Time per model: {time_per_model} minutes")
    print(f"Calculated walltime: {walltime}")

    # Build mail directives
    mail_directives = ""
    if mail_user:
        mail_directives = f"""#SBATCH --mail-type=ALL
#SBATCH --mail-user={mail_user}"""

    # Build the script
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={new_grid_name[:16]}  # Truncated to 16 chars",
        f"#SBATCH --array={array_range}",
        f"#SBATCH -p {partition}",
        f"#SBATCH -A {account}",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH -t {walltime}",
        "#SBATCH --mem=4G",
    ]

    if mail_directives:
        script_lines.append(mail_directives)

    script_lines.extend([
        "#SBATCH --output=./logs/log%A_%a.out",
        "#SBATCH --error=./logs/log%A_%a.err",
        "",
        f"# Grid: {new_grid_name}",
        f"TOTAL_SAMPLES={n_samples}",
        f"GRID_DIR=\"{grid_dir}\"",
        f"CLOUDY_EXE=\"{cloudy_executable_path}/cloudy.exe\"",
        "",
        "task_id=${SLURM_ARRAY_TASK_ID}",
        "",
    ])

    if use_striding_actual:
        script_lines.extend([
            f"STEP={stride_step}",
            "",
            "# Process samples using striding pattern",
            "for incident_idx in $(seq $task_id $STEP $((TOTAL_SAMPLES - 1))); do",
            '    sample_dir="$GRID_DIR/${incident_idx}"',
            "",
            '    if [ ! -d "$sample_dir" ]; then',
            '        echo "Warning: Directory $sample_dir not found"',
            "        continue",
            "    fi",
            "",
            '    cd "$sample_dir"',
            '    echo "Processing sample $incident_idx in $PWD"',
            "",
        ])
    else:
        script_lines.extend([
            "incident_idx=$task_id",
            'sample_dir="$GRID_DIR/${incident_idx}"',
            "",
            'if [ ! -d "$sample_dir" ]; then',
            '    echo "Error: Directory $sample_dir not found"',
            "    exit 1",
            "fi",
            "",
            'cd "$sample_dir"',
            'echo "Processing sample $incident_idx in $PWD"',
            "",
        ])

    # Run Cloudy models sequentially
    no_files_action = "        continue" if use_striding_actual else "        exit 1"
    script_lines.extend([
        "    # Run Cloudy models sequentially",
        '    num_files=$(ls *.in 2>/dev/null | wc -l)',
        "",
        '    if [ "$num_files" -eq 0 ]; then',
        '        echo "No .in files found in $sample_dir"',
        no_files_action,
        "    fi",
        "",
        "    for model_idx in $(seq 0 $((num_files - 1))); do",
        '        echo "Running cloudy.exe -r $model_idx"',
        "        $CLOUDY_EXE -r $model_idx",
        "    done",
    ])

    if use_striding_actual:
        script_lines.append("done")

    slurm_job_script = "\n".join(script_lines) + "\n"

    # Save job script
    script_filename = f"{cloudy_output_dir}/{new_grid_name}/array.slurm"
    with open(script_filename, "w") as f:
        f.write(slurm_job_script)

    print(f"Created SLURM job script: {script_filename}")
    return script_filename


def cosma7_sobol_onthefly(
    new_grid_name,
    n_samples,
    cloudy_output_dir,
    cloudy_executable_path,
    python_env=None,
    account="dp004",
    partition="cosma7",
    time_per_model=10,
    use_striding=True,
    stride_step=1000,
    mail_user=None,
):
    """
    Create COSMA7 SLURM script for Sobol grids with on-the-fly input generation.
    Generates inputs, runs Cloudy, extracts spectra, and cleans up.
    """
    from pathlib import Path

    grid_dir = f"{cloudy_output_dir}/{new_grid_name}"
    use_striding_actual = use_striding and n_samples > stride_step

    if use_striding_actual:
        n_array_jobs = min(stride_step, n_samples)
        array_range = f"0-{n_array_jobs - 1}"
        models_per_job = (n_samples + stride_step - 1) // stride_step
    else:
        array_range = f"0-{n_samples - 1}"
        models_per_job = 1

    total_minutes = int(models_per_job * time_per_model * 1.2)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    walltime = f"{hours:02d}:{minutes:02d}:00"

    print(f"Models per array job: {models_per_job}")
    print(f"Calculated walltime: {walltime}")

    mail_directives = ""
    if mail_user:
        mail_directives = f"""#SBATCH --mail-type=ALL
#SBATCH --mail-user={mail_user}"""

    script_dir = str(Path(__file__).parent.absolute())

    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={new_grid_name[:16]}",
        f"#SBATCH --array={array_range}",
        f"#SBATCH -p {partition}",
        f"#SBATCH -A {account}",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH -t {walltime}",
        "#SBATCH --mem=4G",
    ]

    if mail_directives:
        script_lines.append(mail_directives)

    script_lines.extend([
        "#SBATCH --output=./logs/log%A_%a.out",
        "#SBATCH --error=./logs/log%A_%a.err",
        "",
        f"TOTAL_SAMPLES={n_samples}",
        f"GRID_DIR=\"{grid_dir}\"",
        f"CLOUDY_EXE=\"{cloudy_executable_path}/cloudy.exe\"",
        f"SCRIPT_DIR=\"{script_dir}\"",
        "",
        "# Create spectra output directory",
        'mkdir -p "$GRID_DIR/spectra"',
        "",
    ])

    if python_env:
        script_lines.extend([f"source {python_env}/bin/activate", ""])

    script_lines.append("task_id=${SLURM_ARRAY_TASK_ID}")
    script_lines.append("")

    if use_striding_actual:
        script_lines.extend([
            f"STEP={stride_step}",
            "",
            "for incident_idx in $(seq $task_id $STEP $((TOTAL_SAMPLES - 1))); do",
            '    work_dir="$GRID_DIR/tmp_${incident_idx}_$$"',
            '    mkdir -p "$work_dir"',
            '    echo "Sample $incident_idx"',
            '    python "$SCRIPT_DIR/generate_cloudy_input_onthefly.py" --grid-dir "$GRID_DIR" --sample-index $incident_idx --work-dir "$work_dir" || continue',
            '    cd "$work_dir"',
            "    $CLOUDY_EXE -r 0",
            '    if [ $? -eq 0 ] && [ -f "0.cont" ]; then',
            '        tail -n +2 0.cont | awk \'{print $1, $3, $4, $7}\' > "$GRID_DIR/spectra/spectra_${incident_idx}.txt"',
            '        cd "$GRID_DIR" && rm -rf "$work_dir"',
            "    fi",
            "done",
        ])
    else:
        script_lines.extend([
            "incident_idx=$task_id",
            'work_dir="$GRID_DIR/tmp_${incident_idx}_$$"',
            'mkdir -p "$work_dir"',
            'python "$SCRIPT_DIR/generate_cloudy_input_onthefly.py" --grid-dir "$GRID_DIR" --sample-index $incident_idx --work-dir "$work_dir" || exit 1',
            'cd "$work_dir"',
            "$CLOUDY_EXE -r 0",
            'if [ $? -eq 0 ] && [ -f "0.cont" ]; then',
            '    tail -n +2 0.cont | awk \'{print $1, $3, $4, $7}\' > "$GRID_DIR/spectra/spectra_${incident_idx}.txt"',
            '    cd "$GRID_DIR" && rm -rf "$work_dir"',
            "fi",
        ])

    slurm_job_script = "\n".join(script_lines) + "\n"

    script_filename = f"{cloudy_output_dir}/{new_grid_name}/array.slurm"
    with open(script_filename, "w") as f:
        f.write(slurm_job_script)

    print(f"Created on-the-fly SLURM script: {script_filename}")
    return script_filename

