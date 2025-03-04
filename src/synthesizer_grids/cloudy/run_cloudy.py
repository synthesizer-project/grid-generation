"""
Run cloudy.

Depending on the choice of --incident-index and --photoionisation-index it is
possible to run either a single model (setting both), all models for a given
incident grid point (setting only --incident-index, the recommended approach),
or all models (setting neither).
"""

import argparse
import os

import yaml

if __name__ == "__main__":

    # Initial parser
    parser = argparse.ArgumentParser(
        description="Run the cloudy models",)

    # Add additional parameters which are specific to this script

    parser.add_argument(
        "--cloudy-output-dir",
        type=str,
        required=False,
        default=None)

    parser.add_argument(
        "--grid-name",
        type=str,
        required=False,
        default=None)

    # Path to cloudy directory (not the executable; this is assumed to
    # {cloudy}/{cloudy_version}/source/cloudy.exe)
    parser.add_argument(
        "--cloudy-executable-path",
        type=str,
        required=False,
        default=None)

    # The incident-index. If not set loop over all incident indices. NOTE:
    # this will be slow and should only be used for small grids. In practice
    # this should be argument set by a HPC array job.
    parser.add_argument(
        "--incident-index",
        type=int,
        required=False,
        default=None)

    # The photoionisation-index. If not set loop over all
    # photoionisation-indicies indices. By default this should be None so
    # that each call to run_cloudy.py loops over all photoionisation models.
    parser.add_argument(
        "--photoionisation-index",
        type=int,
        required=False,
        default=None)

    # Parse arguments
    args = parser.parse_args()

    # Shorthand for cloudy_output_dir
    output_directory = f"{args.cloudy_output_dir}/{args.grid_name}"

    # Open the file containing the details of the photoionisation model
    parameter_file = f"{output_directory}/grid_parameters.yaml"
    with open(parameter_file, "r") as file:
        parameters = yaml.safe_load(file)

    # set CLOUDY_DATA_PATH environment variable
    os.environ['CLOUDY_DATA_PATH'] = (
        f"{args.cloudy_executable_path}/"
        f"{parameters['cloudy_version']}/data/:./")

    # Set incident_indices
    # If an index is provided just use that
    if args.incident_index is not None:
        incident_indices = [args.incident_index]
    # Other wise use a range
    else:
        incident_indices = range(parameters['incident_n_models'])

    # Set photoionisation_indices
    # If an index is provided just use that
    if args.photoionisation_index is not None:
        photoionisation_indices = [args.photoionisation_index]
    # Other wise use a range
    else:
        photoionisation_indices = range(parameters['photoionisation_n_models'])

    for incident_index in incident_indices:

        # change directory to the output directory
        os.chdir(f"{output_directory}/{incident_index}")

        # Loop over each photoionisation model
        for photoionisation_index in photoionisation_indices:

            input_file = (
                f"{output_directory}/{incident_index}"
                f"/{photoionisation_index}.in"
                )

            cloudy_executable = (
                f"{args.cloudy_executable_path}/{parameters['cloudy_version']}"
                "/source/cloudy.exe")

            # Run the cloudy job
            command = f'{cloudy_executable} -r {photoionisation_index}'
            print(command)
            os.system(command)
