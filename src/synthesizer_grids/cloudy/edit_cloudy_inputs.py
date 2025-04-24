"""
This is a simple script to allow the batch editting of a set of cloudy input
scripts.

line_to_remove = 'iterate to convergence'

"""

import argparse

import numpy as np


def edit_line_by_content(filename, line_to_edit, replacement=False):
    with open(filename, "r") as file:
        lines = file.readlines()

    with open(filename, "w") as file:
        for line in lines:
            if line.strip() == line_to_edit:
                if replacement:
                    file.write(replacement + "/n")
            else:
                file.write(line)


if __name__ == "__main__":
    # Initial parser
    parser = argparse.ArgumentParser(
        description="Edit cloudy input files",
    )

    # Add additional parameters which are specific to this script

    # The cloudy output directory
    parser.add_argument(
        "--cloudy-output-dir",
        type=str,
        required=False,
        default=None,
    )

    # The full grid name
    parser.add_argument(
        "--grid-name",
        type=str,
        required=False,
        default=None,
    )

    # The filename of a list of incident and photoionisation indices to edit.
    parser.add_argument(
        "--list-file",
        type=str,
        required=False,
        default=None,
    )

    # The index in the list to run if only one model is to be changed.
    parser.add_argument(
        "--list-index",
        type=int,
        required=False,
        default=None,
    )

    # The line content (not index) to edit
    parser.add_argument(
        "--line_to_edit",
        type=str,
        required=True,
        default=None,
    )

    # Optional replacement, otherwise simply remove
    parser.add_argument(
        "--replacement",
        type=str,
        required=False,
        default=False,
    )

    # Parse arguments
    args = parser.parse_args()

    # Shorthand for cloudy_output_dir
    output_directory = f"{args.cloudy_output_dir}/{args.grid_name}"

    # If a list is provided open the file containing the list
    if args.list_file is not False:
        incident_indices, photoionisation_indices = np.loadtxt(
            args.list_file, dtype=int
        )

        # If an index is also provided just choose this model
        if args.list_index is not False:
            incident_indices = [incident_indices[args.list_index]]
            photoionisation_indices = [
                photoionisation_indices[args.list_index]
            ]

    print(incident_indices)
    print(photoionisation_indices)

    # Loop over the list of indices and run each one
    for incident_index, photoionisation_index in zip(
        incident_indices, photoionisation_indices
    ):
        # The file to edit
        filename = (
            f"{output_directory}/{incident_index}/{photoionisation_index}.in"
        )

        print(incident_index, photoionisation_index, filename)

        # Edit the file
        edit_line_by_content(filename, args.line_to_edit, args.replacement)
