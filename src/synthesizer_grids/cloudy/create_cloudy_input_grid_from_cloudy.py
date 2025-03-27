"""
Create a grid of cloudy scripts based on a cloudy in-built incident model such
as a blackbody or AGN.

NOTE: this will not currently work and this functionality (generating grids
directly from cloudy) may be removed in the future. Much simpler to just use
incident grids.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

# synthesiser modules
from synthesizer.abundances import Abundances
from synthesizer.photoionisation import cloudy17, cloudy23

# local modules
from utils import apollo_submission_script, get_grid_properties


class CloudyIncidentShapeCommands:
    """
    A class holding different cloudy incident models. These return the relevant
    shape commands.
    """

    def blackbody(temperature=None, model=None):
        """
        A function for specifying the cloudy blackbody model.

        Args:
            temperature (float)
                The characteristic temperature of the blackbody.

        Returns:
            shape_commands (list, str)
                A list of strings with the cloudy input commands
        """

        shape_commands = []
        shape_commands.append(f"black {temperature} \n")

        return shape_commands

    def agn(
        big_bump_temperature=None, aox=-1.4, auv=-0.5, ax=-1.35, model=None
    ):
        """
        A function for specifying the cloudy AGN model. See 6.2 Hazy1.pdf.

        Args:
            big_bump_temperature (float)
                The Big Bump temperature
            aox (float)
                The x-ray slope (default value from Calabro CEERS AGN model)
            auv (float)
                The uv-slope (default value from Calabro CEERS AGN model)
            ax (float)
                Slope normalisation

        Returns:
            shape_commands (list, str)
                A list of strings with the cloudy input commands
        """

        # collect cloudy shape commands
        shape_commands = []
        shape_commands.append(
            f"AGN T = {big_bump_temperature} k, a(ox) = {aox}, a(uv)= {auv} \
                                a(x)={ax} \n"
        )

        return shape_commands


def load_grid_params(param_file="c17.03-sps", dir="params"):
    """
    Load parameters from a single param_file. This separates parameters into
    those that vary and those that are fixed.

    Parameters
    ----------
    param_file : str
        Location of YAML file.

    Returns
    -------
    dict
        Dictionary of cloudy parameters
    """

    # open paramter file
    with open(f"{dir}/{param_file}.yaml", "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    grid_params = {}
    fixed_params = {}

    for k, v in params.items():
        if isinstance(v, list):
            grid_params[k] = np.array(list(map(float, v)))
        else:
            fixed_params[k] = v

    return fixed_params, grid_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a grid of incident cloudy models"
    )

    # Machine (for submission script generation)
    parser.add_argument("-machine", type=str, required=True)

    # Path to grid directory (i.e. where incident and new grids are stored)
    parser.add_argument("-grid_dir", type=str, required=True)

    # Path to directory where cloudy runs are
    parser.add_argument("-cloudy_dir", type=str, required=True)

    # The name of the file denoting the cloudy model and the variable
    # parameters
    parser.add_argument("-incident_cloudy_model", type=str, required=True)

    # the cloudy parameters, including any grid axes
    parser.add_argument(
        "-cloudy_params", type=str, required=False, default="c17.03-sps"
    )

    # Path to cloudy directory (not executable; this is assumed to
    # {cloudy}/{cloudy_version}/source/cloudy.ext)
    parser.add_argument("-cloudy_path", type=str, required=True)

    # Verbosity flag
    parser.add_argument("-verbose", type=bool, required=False, default=True)

    args = parser.parse_args()

    verbose = args.verbose

    # Load the cloudy parameters you are going to run
    cloudy_fixed_params, cloudy_grid_params = load_grid_params(
        args.cloudy_params
    )

    # Load the incident parameters
    incident_fixed_params, incident_grid_params = load_grid_params(
        args.incident_cloudy_model, dir="cloudy_incident_models"
    )

    print(incident_fixed_params, incident_grid_params)

    # set cloudy version
    if cloudy_fixed_params["cloudy_version"] == "c17.03":
        cloudy = cloudy17
    if cloudy_fixed_params["cloudy_version"] == "c23.01":
        cloudy = cloudy23
    print(cloudy)

    # get name of new grid (concatenation of incident_grid and cloudy
    # parameter file)
    new_grid_name = f"{args.incident_cloudy_model}_cloudy-{args.cloudy_params}"

    print(new_grid_name)

    # Define output directories
    output_dir = f"{args.cloudy_dir}/{new_grid_name}"

    # make output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # for submission system output files
    Path(f"{output_dir}/output").mkdir(parents=True, exist_ok=True)

    # set a list of the axes
    axes = list(incident_grid_params.keys()) + list(cloudy_grid_params.keys())
    if verbose:
        print("axes:", axes)

    # the parameters used to define the incident emission
    incident_parameters = incident_grid_params | incident_fixed_params

    # parameters varied on the grid
    grid_parameters = incident_grid_params | cloudy_grid_params

    # fixed parameters
    fixed_parameters = incident_fixed_params | cloudy_fixed_params

    if verbose:
        # print fixed parameters
        for k, v in fixed_parameters.items():
            print(k, v)

        # print grid parameters, including incident parameters
        for k, v in grid_parameters.items():
            print(k, v)

    # combine all parameters
    params = fixed_parameters | grid_parameters

    # save all parameters
    yaml.dump(params, open(f"{output_dir}/params.yaml", "w"))

    # get properties of the grid
    (
        n_axes,
        shape,
        n_models,
        mesh,
        model_list,
        index_list,
    ) = get_grid_properties(axes, grid_parameters, verbose=True)

    # Create new synthesizer grid to contain the new grid

    # open the new grid
    with h5py.File(f"{args.grid_dir}/{new_grid_name}.hdf5", "w") as hf:
        # add attribute with full grid axes
        hf.attrs["axes"] = axes

        # add the bin centres for the grid bins
        for axis in axes:
            hf[f"axes/{axis}"] = grid_parameters[axis]

        # add other parameters as attributes
        for k, v in params.items():
            # If v is None then convert to string None for saving in the
            # HDF5 file.
            if v is None:
                v = "None"

            # if the parameter is a dictionary (e.g. as used for abundances)
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    hf.attrs[k + "_" + k2] = v2
            else:
                hf.attrs[k] = v

        if verbose:
            print("-" * 50)
            print("---- attributes")
            for k, v in hf.attrs.items():
                print(k, v)
            print("---- groups and datasets")
            hf.visit(print)

    # loop over all models
    for i, (grid_params_tuple, grid_index_tuple) in enumerate(
        zip(model_list, index_list)
    ):
        # get a dictionary of all parameters that are varied over the grid
        grid_params_ = dict(zip(axes, grid_params_tuple))

        print(grid_params_)

        # Get all paramters
        params_ = fixed_parameters | grid_params_

        print(params_)

        # get a dictionary of the parameter grid point
        grid_index_ = dict(zip(axes, grid_index_tuple))

        # Get a dictionary of just the incident parameters. These are the
        # parameters that cloudy uses to generate the incident emission.
        # These can be fixed or not.
        incident_params_ = {k: params_[k] for k in incident_parameters.keys()}

        print(incident_params_)

        abundances = Abundances(
            metallicity=float(params_["metallicity"]),
            reference=params_["reference_abundance"],
            alpha=params_["alpha"],
            abundances=params_["abundance_scalings"],
            depletion_model=params_["depletion_model"],
            depletion_scale=params_["depletion_scale"],
        )

        # For cloudy based grids it makes most sense to assume a fixed
        # ionisation parameter. This will also simplify the code.

        # float() is needed if the ionisation parameter is provided as
        # e.g. 1E1 instead of 1e+1
        params_["ionisation_parameter"] = float(
            params_["ionisation_parameter"]
        )

        # This returns the relevant shape commands, in this case for a
        # tabulated SED

        shape_command_function = getattr(
            CloudyIncidentShapeCommands, incident_params_["model"]
        )

        shape_commands = shape_command_function(**incident_params_)

        # create cloudy input file
        cloudy.create_cloudy_input(
            str(i + 1),
            shape_commands,
            abundances,
            output_dir=output_dir,
            **params_,
        )

    # create submission script
    if args.machine == "apollo":
        apollo_submission_script(
            n_models, output_dir, args.cloudy_path, params_["cloudy_version"]
        )
