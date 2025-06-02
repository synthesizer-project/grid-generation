"""
Check the cloudy runs for failed/successful models
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import re
from collections import namedtuple
from typing import Dict

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from synthesizer.grid import Grid
from tqdm import tqdm

from synthesizer_grids.cloudy.utils import (
    get_cloudy_params,
    get_grid_props_cloudy,
)
from synthesizer_grids.parser import Parser


def read_model_output_tail(
    model_num: int, folder_loc: str, tail_len: int = 5
) -> str:
    """
    Reads in the tail strings of '.out' inside 'folder_name'

    Parameters:
        model_num (int): the number assigned to the cloudy model run
        folder_loc (str): A string containing the directory with cloudy outputs
        tail_len (int): Number of tail strings to read in

    Returns:
        A string with the last 'tail_len' lines of output, returns 0 if
        file not found
    """

    try:
        with open(f"{folder_loc}/{model_num}.out", "r") as f:
            tail = f.readlines()[-tail_len:]
        tail = "".join(tail)
        # Remove all newline (\n) and square brackets
        tail = re.sub(r"[\n]|\[|\]", "", tail)
    except FileNotFoundError:
        if os.path.exists(f"{folder_loc}/{model_num}.in"):
            tail = None
        else:
            sys.exit(
                f"File {folder_loc}/{model_num} does not exist!"
                "You might be looking in the wrong folder!"
            )

    return tail


def check_run(
    args: namedtuple,
    variable_photoionisation_params: Dict,
    all_model_params_unit: Dict,
    all_model_params_logged: Dict,
    colourmap: cmr = cmr.tropical,
) -> None:
    """
    Reads in the cloudy runs and checks the output for failed runs

    Parameters:
        args (NamedSpace):
            parser arguments passed on to this job
        variable_photoionisation_params (Dict):
            photoinisation parameters in cloudy that vary
        all_model_params_unit (Dict):
            all the varying parameter units
        all_model_params_logged (Dict):
            units for the varying parameters
        colourmap (matplotlib):
            matplotlib colourmap to use
    """

    # Get properties of the grid
    (
        photoionisation_n_axes,
        photoionisation_shape,
        photoionisation_n_models,
        photoionisation_mesh,
        photoionisation_model_list,
        photoionisation_index_list,
    ) = get_grid_props_cloudy(
        variable_photoionisation_params.keys(),
        variable_photoionisation_params,
        verbose=args.verbose,
    )

    # Open the incident grid
    incident_grid = Grid(
        args.incident_grid + ".hdf5",
        grid_dir=f"{args.grid_dir}",
        read_lines=False,
    )
    # Extract axes and axes values from the Grid
    incident_axes = incident_grid.axes
    incident_axes_values = incident_grid._axes_values

    # Get properties of the incident grid
    (
        incident_n_axes,
        incident_shape,
        incident_n_models,
        incident_mesh,
        incident_model_list,
        incident_index_list,
    ) = get_grid_props_cloudy(
        incident_axes, incident_axes_values, verbose=args.verbose
    )

    # Set up the total model list including both the
    # the incident and photionisation parameters
    all_model_params = incident_axes_values | variable_photoionisation_params
    lengths = [len(value) for value in all_model_params.values()]
    total_models = np.prod(lengths)
    all_model_list = np.zeros((total_models, len(all_model_params)))

    # Fill array with the incident models
    arr = np.repeat(
        incident_model_list, repeats=photoionisation_n_models, axis=0
    )
    all_model_list[:, :incident_n_axes] = arr

    # Fill array with the photoionisation models
    arr = np.tile(photoionisation_model_list, (incident_n_models, 1))
    all_model_list[:, incident_n_axes:] = arr

    # Define dictionary to store summary statistics
    run_space = np.zeros(total_models, dtype=int)
    run_key = {
        0: "Success",
        1: "DNR",  # did not run (could be because of scheduling)
        2: "Unphysical",  # problem with parameter space or negative population
        3: "Converge",  # did not converge
        4: "Abort",  # cloudy aborted
        5: "Wrong",  # something went wrong, like hitting some default limit
        # before convergence, such as the number of zones
        6: "Empty",  # cloudy input file issue
        7: "DNF",  # did not finish in time
    }

    # Accounting array
    outcome_array = np.zeros(len(run_key))

    # Variable to run through all models
    ii = 0

    print(f"\nReading model outcomes for all models in {args.cloudy_dir}")

    for incident_index in tqdm(range(incident_n_models)):
        for photoionisation_index in range(photoionisation_n_models):
            # Get the tail of the cloudy output file
            tail = read_model_output_tail(
                model_num=photoionisation_index,
                folder_loc=f"{args.cloudy_dir}/{incident_index}/",
            )

            if "Cloudy exited OK" in tail:
                outcome_array[0] += 1
            elif tail is None:
                run_space[ii] = 1
                outcome_array[1] += 1
            elif (
                "unphysical" in tail
                or "negative population" in tail
                or "start" in tail
            ):
                run_space[ii] = 2
                outcome_array[2] += 1
            elif "did not converge" in tail:
                run_space[ii] = 3
                outcome_array[3] += 1
            elif "ABORT" in tail:
                run_space[ii] = 4
                outcome_array[4] += 1
            elif "something went wrong" in tail:
                run_space[ii] = 5
                outcome_array[5] += 1
            elif tail == "":
                run_space[ii] = 6
                outcome_array[6] += 1
            elif "Cloudy exited OK" not in tail:
                run_space[ii] = 7
                outcome_array[7] += 1

            ii += 1

    print(f"\n{total_models} models in total in this sample")
    print("Breakdown of the EXIT codes:\n")
    for ii in range(len(run_key)):
        outcome_percent = np.round(100 * outcome_array[ii] / total_models, 3)
        print(f"\t{outcome_percent}% \t|\t{ii}: {run_key[ii]} ")

    if outcome_array[0] != total_models:
        print(
            f"{int(total_models - outcome_array[0])} runs were not successful"
        )
        print(
            f"Saving to run dictionary to {args.cloudy_dir}/"
            "run_space_outcome.npz file\n"
        )
        np.savez(f"{args.cloudy_dir}/run_space_outcome", data=run_space)
        print("Creating parameter space vs run outcome visualisation\n")
        plot_run_space(
            args,
            all_model_params,
            all_model_list,
            all_model_params_unit,
            all_model_params_logged,
            run_space,
            run_key,
            colourmap=colourmap,
        )
    else:
        print("All models ran successfully!")


def plot_run_space(
    args: namedtuple,
    parameters: Dict,
    model_list: NDArray,
    param_unit: Dict,
    param_logged: Dict,
    run_space: NDArray,
    run_key: Dict,
    colourmap: cmr,
) -> None:
    """
    Plots the parameter space of successful and failed
    cloudy runs as a scatter matrix

    Parameters:
        args (NamedSpace):
            parser arguments passed on to this job
        parameters (Dict):
            parameters in the cloudy runs that vary
        model_list (NDArray):
            A N-D array of model parameters
        param_unit (Dict):
            all the varying parameter units
        param_logged (Dict):
            units for the varying parameters
        run_space (NDArray):
            the exit code for the parameter space,
            indicating success or failure (and what
            kind of failure)
        run_key (Dict):
            the different defined exit codes
        colourmap (matplotlib):
            matplotlib colourmap to use
    """

    # How many parameters
    N = len(parameters.keys())

    # Set up parameter ranges and labels
    p_ranges = {}
    p_labels = {}
    for key, value in parameters.items():
        if param_logged[key]:
            p_ranges[key] = [
                np.min(value) - 0.2,
                np.max(value) + 0.2,
            ]
        else:
            p_ranges[key] = [
                np.min(np.log10(value)) - 0.2,
                np.max(np.log10(value)) + 0.2,
            ]
        if param_unit[key] == "None":
            p_labels[key] = f"{key}"
        else:
            p_labels[key] = f"{key}/{param_unit[key]}"

    # Some plot settings
    marker_size = 20
    tick_label_size = 8
    label_size = 10

    # Some run space manipulation for separate plotting
    success = run_space == 0

    run_ticks = np.array(list(run_key.keys()))
    run_labels = np.array(list(run_key.values()))

    # Set up main plot
    f, ax_array = plt.subplots(N, N, figsize=(16, 16))

    # Choose a colourmap
    c_m = colourmap
    norm = matplotlib.colors.BoundaryNorm(
        np.arange(-0.5, len(run_ticks) + 0.5), c_m.N
    )

    for ii, key_ii in enumerate(parameters.keys()):
        for jj, key_jj in enumerate(parameters.keys()):
            if param_logged[key_jj]:
                x = model_list[:, jj]
            else:
                x = np.log10(model_list)[:, jj]

            if param_logged[key_ii]:
                y = model_list[:, ii]
            else:
                y = np.log10(model_list)[:, ii]

            # Empty diagonals
            if ii == jj:
                ax_array[ii, jj].axis("off")

            # Successful runs in the upper triangle
            elif jj > ii:
                ax_array[ii, jj].scatter(
                    x=x[success],
                    y=y[success],
                    c=run_space[success],
                    s=marker_size,
                    alpha=1,
                    edgecolors="None",
                    cmap=c_m,
                    norm=norm,
                )

                ax_array[ii, jj].set_ylim(p_ranges[key_ii])
                ax_array[ii, jj].set_xlim(p_ranges[key_jj])

                ax_array[ii, jj].grid(True, ls="dashed")
                ax_array[ii, jj].tick_params(
                    axis="x",
                    which="major",
                    labelsize=tick_label_size,
                    top=True,
                    bottom=False,
                )
                ax_array[ii, jj].tick_params(
                    axis="y",
                    which="major",
                    labelsize=tick_label_size,
                    right=True,
                    left=False,
                )

                if jj == N - 1:
                    ax_array[ii, jj].set_ylabel(
                        ylabel=r"log$_{{10}}$" + f"({p_labels[key_ii]})",
                        size=label_size,
                        labelpad=10,
                    )
                    ax_array[ii, jj].yaxis.set_label_position("right")
                    ax_array[ii, jj].yaxis.set_tick_params(
                        right="on", left="off"
                    )
                    ax_array[ii, jj].yaxis.set_ticks_position("right")
                if ii == 0:
                    ax_array[ii, jj].set_xlabel(
                        xlabel=r"log$_{{10}}$" + f"({p_labels[key_jj]})",
                        size=label_size,
                        labelpad=10,
                    )
                    ax_array[ii, jj].xaxis.set_label_position("top")
                    ax_array[ii, jj].xaxis.set_tick_params(
                        top="on", bottom="off"
                    )
                    ax_array[ii, jj].xaxis.set_ticks_position("top")
                if jj < N - 1:
                    ax_array[ii, jj].set_yticklabels([])
                if ii > 0:
                    ax_array[ii, jj].set_xticklabels([])

                for label in (
                    ax_array[ii, jj].get_xticklabels()
                    + ax_array[ii, jj].get_yticklabels()
                ):
                    label.set_fontsize(12)

            # Plotting the failed runs in the lower triangle
            elif jj < ii:
                ax_array[ii, jj].scatter(
                    x=x[~success] + 0.01 * run_space[~success],
                    y=y[~success] + 0.01 * run_space[~success],
                    c=run_space[~success],
                    s=10 * (run_space[~success] + 1),
                    alpha=0.2,
                    edgecolors="None",
                    cmap=c_m,
                    norm=norm,
                )

                ax_array[ii, jj].set_ylim(p_ranges[key_ii])
                ax_array[ii, jj].set_xlim(p_ranges[key_jj])

                ax_array[ii, jj].grid(True, ls="dashed")
                ax_array[ii, jj].tick_params(
                    axis="x",
                    which="major",
                    labelsize=tick_label_size,
                    top=False,
                )
                ax_array[ii, jj].tick_params(
                    axis="y",
                    which="major",
                    labelsize=tick_label_size,
                    right=False,
                )

                if ii == N - 1:
                    ax_array[ii, jj].set_xlabel(
                        xlabel=r"log$_{{10}}$" + f"({p_labels[key_jj]})",
                        size=label_size,
                        labelpad=10,
                    )
                    ax_array[ii, jj].xaxis.set_label_position("bottom")
                    ax_array[ii, jj].xaxis.set_tick_params(top="off")
                if jj == 0:
                    ax_array[ii, jj].set_ylabel(
                        ylabel=r"log$_{{10}}$" + f"({p_labels[key_ii]})",
                        size=label_size,
                        labelpad=10,
                    )
                    ax_array[ii, jj].yaxis.set_label_position("left")
                    ax_array[ii, jj].yaxis.set_tick_params(right="off")
                if jj > 0:
                    ax_array[ii, jj].set_yticklabels([])
                if ii != N - 1:
                    ax_array[ii, jj].set_xticklabels([])

                for label in (
                    ax_array[ii, jj].get_xticklabels()
                    + ax_array[ii, jj].get_yticklabels()
                ):
                    label.set_fontsize(12)

    f.subplots_adjust(
        hspace=0, wspace=0, left=0.13, bottom=0.10, right=0.9, top=0.98
    )

    # Make space for colorbar
    cbaxes = f.add_axes([0.98, 0.25, 0.02, 0.5])
    f.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=c_m),
        cax=cbaxes,
        orientation="vertical",
    )
    cbaxes.set_yticks(run_ticks)
    cbaxes.set_yticklabels(run_labels, fontsize=10)

    # Build file name and save figure
    file_name = "outcome_vs_p-space.png"
    output_path = os.path.join(args.cloudy_dir, file_name)
    f.savefig(output_path, bbox_inches="tight", dpi=300)

    print(f"Saved plot to: {output_path}")

    plt.close()


if __name__ == "__main__":
    parser = Parser(
        description="Check the parameter space of cloudy models"
        "for success/falure"
    )

    # The name of the incident grid
    parser.add_argument("--incident_grid", type=str, required=True)

    # Path to directory where cloudy runs are
    parser.add_argument("--cloudy_dir", type=str, required=True)

    # The cloudy parameters, including any grid axes
    # This is the parameter file within the cloudy
    # run directory
    parser.add_argument("--cloudy_paramfile", type=str, required=True)

    parser.add_argument(
        "--extra_cloudy_paramfile", type=str, required=False, default=None
    )

    # Parse arguments
    args = parser.parse_args()

    # Unpack arguments
    grid_dir = args.grid_dir
    cloudy_dir = args.cloudy_dir
    verbose = args.verbose

    # Open the incident grid
    incident_grid = Grid(
        args.incident_grid + ".hdf5",
        grid_dir=f"{args.grid_dir}",
        read_lines=False,
    )

    # Load the cloudy parameters you are going to run
    fixed_photoionisation_params, variable_photoionisation_params = (
        get_cloudy_params(args.cloudy_paramfile, param_dir=args.cloudy_dir)
    )

    # If an additional parameter set is provided supersede the default
    # parameters with these.
    if args.extra_cloudy_paramfile is not None:
        (fixed_photoionisation_params_, variable_photoionisation_params_) = (
            get_cloudy_params(args.extra_cloudy_paramfile)
        )
        fixed_photoionisation_params |= fixed_photoionisation_params_
        variable_photoionisation_params |= variable_photoionisation_params_

    if verbose:
        print("axes:", variable_photoionisation_params)

    # If the ionisation_parameter_model is the reference model (i.e. not fixed)
    # save the grid point for the reference values
    if fixed_photoionisation_params["ionisation_parameter_model"] == "ref":
        # Initialize an empty list to store reference values
        reference_values = []

        # Iterate over the axes of the incident grid
        for k in incident_grid.axes:
            # Adjust the name of k as needed
            if k == "metallicities":
                k = "metallicity"
            elif k == "ages":
                k = "age"

            # Append the corresponding reference value from fixed_params
            reference_values.append(
                fixed_photoionisation_params["reference_" + k]
            )

        # Get the reference grid point using the adjusted reference values
        incident_ref_grid_point = incident_grid.get_grid_point(
            reference_values
        )

        # Add the reference grid point indices to fixed_params
        for k, i in zip(incident_grid.axes, incident_ref_grid_point):
            # Adjust the axis names again before saving the index
            if k == "metallicities":
                k = "metallicity"
            elif k == "ages":
                k = "age"

            # Save the index to the fixed_params dictionary
            fixed_photoionisation_params["reference_" + k + "_index"] = i

    # Combine all the varying parameters
    all_model_params = (
        incident_grid._axes_values | variable_photoionisation_params
    )

    all_model_params_unit = {}
    all_model_params_logged = {}
    for key in all_model_params.keys():
        if "age" in key:
            all_model_params_unit[key] = "yr"
            all_model_params_logged[key] = False
        elif "hydrogen_density" in key:
            all_model_params_unit[key] = "cm-3"
            all_model_params_logged[key] = False
        elif "column_density" in key:
            all_model_params_unit[key] = "cm-2"
            all_model_params_logged[key] = True
        elif "turbulence" in key:
            all_model_params_unit[key] = "km/s"
            all_model_params_logged[key] = False
        else:
            all_model_params_unit[key] = "None"
            all_model_params_logged[key] = False

    check_run(
        args,
        variable_photoionisation_params,
        all_model_params_unit,
        all_model_params_logged,
        colourmap=cmr.tropical,
    )
