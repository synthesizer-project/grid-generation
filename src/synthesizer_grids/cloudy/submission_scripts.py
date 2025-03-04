"""
Functions for creating submission scripts for specific machines.
"""


def create_slurm_job_script_by_incident_grid_point(
        number_of_incident_grid_points=None,
        partition=None,
        new_grid_name=None,
        cloudy_output_dir=None,
        cloudy_executable_path=None,
        memory="4G"):

    """
    Create a generic slurm input script where we loop over every
    photoionisation model for a given incident grid point. In this case the
    number of jobs is the number of incident grid points and the number of
    cloudy runs is the number of photoionisation grid points.
    """

    slurm_job_script = f"""#!/bin/bash
#SBATCH --job-name=run_cloudy      # Job name
#SBATCH --output=output/%A_%a.out  # Standard output log
#SBATCH --error=output/%A_%a.err   # Error log
#SBATCH --array=0-{number_of_incident_grid_points-1}  # Job array range
#SBATCH --ntasks=1                 # Number of tasks per job
#SBATCH --cpus-per-task=1          # CPU cores per task
#SBATCH --mem={memory}             # Memory per task
#SBATCH --partition={partition}    # Partition/queue name

# Run command
python run_cloudy.py \\
    --grid-name={new_grid_name} \\
    --cloudy-output-dir={cloudy_output_dir} \\
    --cloudy-executable-path={cloudy_executable_path} \\
    --incident-index=${{SLURM_ARRAY_TASK_ID}}
"""
    return slurm_job_script


def create_slurm_job_script_by_photoionisation_grid_point(
        number_of_photoionisation_models=None,
        partition=None,
        new_grid_name=None,
        cloudy_output_dir=None,
        cloudy_executable_path=None,
        memory="4G"):

    """
    Create a generic slurm input script where we loop over every
    photoionisation model for a given photoionisation grid point. In this
    case the number of jobs is the number of photoionisation grid points and
    the number of cloudy runs per jobs is the number of incident grid points.
    """

    slurm_job_script = f"""#!/bin/bash
#SBATCH --job-name=run_cloudy      # Job name
#SBATCH --output=output/%A_%a.out  # Standard output log
#SBATCH --error=output/%A_%a.err   # Error log
#SBATCH --array=0-{number_of_photoionisation_models-1}  # Job array range
#SBATCH --ntasks=1                 # Number of tasks per job
#SBATCH --cpus-per-task=1          # CPU cores per task
#SBATCH --mem={memory}             # Memory per task
#SBATCH --partition={partition}    # Partition/queue name

# Run command
python run_cloudy.py \\
    --grid-name={new_grid_name} \\
    --cloudy-output-dir={cloudy_output_dir} \\
    --cloudy-executable-path={cloudy_executable_path} \\
    --photoionisation-index=${{SLURM_ARRAY_TASK_ID}}
"""
    return slurm_job_script


def artemis(
    new_grid_name,
    number_of_incident_grid_points=None,
    number_of_photoionisation_models=None,
    cloudy_output_dir=None,
    cloudy_executable_path=None,
    memory="4G",
    by_photoionisation_grid_point=False
        ):

    """
    Submission script generator for artemis
    """

    if by_photoionisation_grid_point:
        number_of_models = number_of_incident_grid_points
        number_of_jobs = number_of_photoionisation_models
    else:
        number_of_models = number_of_photoionisation_models
        number_of_jobs = number_of_incident_grid_points

    # determine the partition to use:
    # short = 2 hours
    if number_of_models < 5:
        partition = 'short'

    # general = 8 hours
    elif number_of_models < 33:
        partition = 'general'

    # long = 8 days
    else:
        partition = 'long'

    # create job script
    if by_photoionisation_grid_point:
        slurm_job_script = (
            create_slurm_job_script_by_photoionisation_grid_point(
                number_of_jobs,
                partition=partition,
                new_grid_name=new_grid_name,
                cloudy_output_dir=cloudy_output_dir,
                cloudy_executable_path=cloudy_executable_path,
                memory=memory))
    else:
        slurm_job_script = create_slurm_job_script_by_incident_grid_point(
            number_of_jobs,
            partition=partition,
            new_grid_name=new_grid_name,
            cloudy_output_dir=cloudy_output_dir,
            cloudy_executable_path=cloudy_executable_path,
            memory=memory)


    # save job script
    open(f"{new_grid_name}.slurm", "w").write(slurm_job_script)
