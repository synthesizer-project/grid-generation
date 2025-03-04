"""
Create a grid of cloudy scripts based on combining an incident grid with a set
of cloudy parameters. Also creates a machine specific script.

Creates a folder for every incident grid point and then input files for every
individual photoionisation model.
"""

import shutil
from pathlib import Path

import numpy as np
import submission_scripts
import yaml
from synthesizer.abundances import (
    Abundances,
)
from synthesizer.exceptions import InconsistentParameter
from synthesizer.grid import Grid
from synthesizer.photoionisation import cloudy17, cloudy23
from utils import (
    get_cloudy_params,
    get_grid_props_cloudy,
)

from synthesizer_grids.parser import Parser


def create_cloudy_input(
        incident_index,
        photoionisation_index,
        parameters,
        delta_log10_specific_ionising_luminosity,
        output_directory,
        ):
    """
    Function that creates a cloudy input script for a single photoionisation
    model.

    Args:
        incident_index (int):
            The index of the incident grid point.
        photoionisation_index (str):
            The index of the photoionisation grid point.
        parameters (dict):
            Dictionary of parameters.
        delta_log10_specific_ionising_luminosity (float):
            The difference between the reference specific ionising luminosity
            and that at the current grid point. For the reference model this
            is used to scale the ionisation parameter and ionising luminosity
            of the source.
        output_directory (str):
            The output directory.
    """

    # only add 'abundance_scalings' if its needed
    if 'abundance_scalings' not in list(parameters.keys()):
        parameters['abundance_scalings'] = {}

    for k, v in parameters.items():
        if len(k.split('.')) > 1:
            if k.split('.')[0] == 'abundance_scalings':
                kk = k.split('.')[1]
                # convert to synthesizer standard
                parameters['abundance_scalings'][kk] = v

    # Create synthesizer.Abundance object
    abundances = Abundances(

        # so horrible
        metallicity=float(parameters["metallicities"]),
        reference=parameters["reference_abundance"],
        alpha=parameters["alpha"],
        abundances=parameters["abundance_scalings"],
        depletion_model=parameters["depletion_model"],
        depletion_scale=parameters["depletion_scale"],
    )

    # Define the ionisation parameter and geometry

    # If the reference ionionisation parameter model is used
    if parameters["ionisation_parameter_model"] == "ref":

        # For spherical geometry the effective log10U is this
        if parameters["geometry"] == "spherical":
            ionisation_parameter = 10**(
                np.log10(parameters["reference_ionisation_parameter"])
                + (1 / 3) * delta_log10_specific_ionising_luminosity
            )

        # For spherical geometry the effective log10U is this.
        # The spherical-U model uses U as a direct input to cloudy
        # instead of calculating Q.
        elif parameters["geometry"] == "spherical-U":
            ionisation_parameter = 10**(
                np.log10(parameters["reference_ionisation_parameter"])
                + (1 / 3) * delta_log10_specific_ionising_luminosity
            )

        # For plane-parallel geometry the effective just scales with
        # log10_specific_ionising_luminosity
        elif parameters["geometry"] == "planeparallel":
            ionisation_parameter = 10**(
                np.log10(parameters["reference_ionisation_parameter"])
                + delta_log10_specific_ionising_luminosity
            )

        # If the geometry is not recognised raise an exception
        else:
            raise InconsistentParameter(
                f"""ERROR: do not understand geometry choice:
                {parameters['geometry']}""")

    # If fixed ionisation parameter model is used:
    elif parameters["ionisation_parameter_model"] == "fixed":
        ionisation_parameter = parameters["ionisation_parameter"]

    # If the model is not regnoised raise an exception
    else:
        raise InconsistentParameter(
                f"""ERROR: do not understand U model choice:
            {parameters['ionisation_parameter_model']}""")

    # Set ionisation_parameter to provide cloudy
    parameters["ionisation_parameter"] = ionisation_parameter

    # Convert numpy arrays to lists for use in YAML file
    parameters_to_save = {}
    for key, value in parameters.items():
        if type(value) is np.float64:
            parameters_to_save[key] = float(value)
        else:
            parameters_to_save[key] = value

    # Output dictionary of all parameters as a YAML file

    yaml_filename = f"{output_directory}/{incident_index}/" + \
        f"{photoionisation_index}.yaml"
    with open(yaml_filename, "w") as file:
        yaml.dump(parameters_to_save, file, default_flow_style=False)

    # Include shape command to read SED
    shape_commands = ['table SED "input.sed" \n']

    # Create cloudy input file
    cloudy.create_cloudy_input(
        str(photoionisation_index),
        shape_commands,
        abundances,
        output_dir=f"{output_directory}/{incident_index}",
        **parameters,
    )


if __name__ == "__main__":

    parser = Parser(
        description="Run a grid of incident cloudy models",
        cloudy_args=True,
        )

    # Add additional parameters which are specific to this script

    # Machine (for submission script generation)
    parser.add_argument(
        "--machine",
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

    # By default the job script creates a job for each incident grid points.
    # Including this flag makes a job for each photoionisation grid point,
    # i.e. the number of cloudy runs per job is the number of incident grid
    # points. In most cases this is not the best approach, but for very large
    # photoionisation grids it may be,
    parser.add_argument(
        "--by-photoionisation-grid-point",
        action="store_true",
        default=False)

    # Parse arguments
    args = parser.parse_args()

    incident_grid_name = args.incident_grid
    incident_grid_dir = args.grid_dir
    cloudy_output_dir = args.cloudy_output_dir
    cloudy_paramfile = args.cloudy_paramfile
    extra_cloudy_paramfile = args.cloudy_paramfile_extra
    machine = args.machine
    cloudy_executable_path = args.cloudy_executable_path
    by_photoionisation_grid_point = args.by_photoionisation_grid_point

    print(incident_grid_name)
    print(incident_grid_dir)
    print(cloudy_output_dir)
    print(cloudy_paramfile)
    print(extra_cloudy_paramfile)

    # Get name of new grid (concatenation of incident_grid and cloudy
    # parameter file)
    new_grid_name = f"{incident_grid_name}_cloudy-{cloudy_paramfile}"

    # If an additional parameter set append this to the new grid name
    if extra_cloudy_paramfile:
        # Ignore the directory part if it exists
        extra_cloudy_paramfile_name = extra_cloudy_paramfile.split("/")[
            -1
        ]
        # Append the new_grid_name with the additional name
        new_grid_name += "-" + extra_cloudy_paramfile_name

    # Check for extensions
    if cloudy_paramfile.split(".")[-1] != "yaml":
        cloudy_paramfile += ".yaml"
    if extra_cloudy_paramfile is not None:
        if extra_cloudy_paramfile.split(".")[-1] != "yaml":
            extra_cloudy_paramfile += ".yaml"
    if incident_grid_name.split(".")[-1] != "hdf5":
        incident_grid_name += ".hdf5"

    # Open the incident grid using synthesizer
    incident_grid = Grid(
        incident_grid_name,
        grid_dir=incident_grid_dir,
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
        incident_axes,
        incident_axes_values,
        verbose=True)

    # Load the cloudy parameters you are going to run
    fixed_photoionisation_params, variable_photoionisation_params = (
        get_cloudy_params(cloudy_paramfile))

    # If an additional parameter set is provided supersede the default
    # parameters with these.
    if extra_cloudy_paramfile:
        (photoionisation_fixed_params_, photoionisation_variable_params_) = (
            get_cloudy_params(extra_cloudy_paramfile)
        )
        fixed_photoionisation_params |= photoionisation_fixed_params_
        variable_photoionisation_params |= photoionisation_variable_params_

    print(fixed_photoionisation_params)
    print(variable_photoionisation_params)

    # If we have photoionisation parameters that vary we need to calculate the
    # model list
    if len(variable_photoionisation_params) > 0:
        # doesn't matter about the ordering of these
        photoionisation_axes = list(variable_photoionisation_params.keys())

        # get properties of the photoionsation grid
        (
            photoionisation_n_axes,
            photoionisation_shape,
            photoionisation_n_models,
            photoionisation_mesh,
            photoionisation_model_list,
            photoionisation_index_list,
        ) = get_grid_props_cloudy(
            photoionisation_axes,
            variable_photoionisation_params,
            verbose=True)

    # Else, we still need to record the number of photoionisation models since
    # this is saved in the YAML file
    else:
        photoionisation_n_models = 1

    # Define output directory
    output_directory = f'{cloudy_output_dir}/{new_grid_name}'

    # Make output directories
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Loop over all incident models, extract the spectral energy distribtions,
    # convert to the cloudy format, and save in the cloudy folder.
    lam = incident_grid.lam

    # Use cloudy version to initialise a synthesizer.cloudy object
    if fixed_photoionisation_params["cloudy_version"] == "c17.03":
        cloudy = cloudy17
    if fixed_photoionisation_params["cloudy_version"] == "c23.01":
        cloudy = cloudy23

    # Save all the parameters for use later
    parameters_to_save = (
        fixed_photoionisation_params |
        variable_photoionisation_params |
        incident_axes_values)
    parameters_to_save = (
        parameters_to_save |
        {'incident_n_models': int(incident_n_models),
         'photoionisation_n_models': int(photoionisation_n_models)})

    # Convert numpy arrays to lists for use in YAML file
    for key, value in parameters_to_save.items():
        if type(value) is np.ndarray:
            parameters_to_save[key] = value.tolist()

    # Save parameters_to_save dictionary to a YAML file
    with open(f"{output_directory}/grid_parameters.yaml", "w") as file:
        yaml.dump(parameters_to_save, file, default_flow_style=False)

    # If the ionisation_parameter_model is the reference model (i.e. not fixed)
    # save the value of the ionising photon luminosity at the reference grid
    # point.
    if fixed_photoionisation_params["ionisation_parameter_model"] == "ref":

        # Initialize an empty list to store reference values
        reference_values = []

        # Iterate over the axes of the incident grid
        for k in incident_grid.axes:

            # We should throw an exception here if a reference value is not
            # included. This will happen for grids with additional axes.

            # In the parameter file we have a reference age and metallicity
            # but the grid axes are metallicities and ages
            if k == 'ages':
                k = 'age'
            if k == 'metallicities':
                k = 'metallicity'

            # Append the corresponding reference value from fixed_params
            reference_values.append(
                fixed_photoionisation_params["reference_"+k])

        # Make dictionary to allow unpacking when getting the reference
        # grid point
        ref_dict = dict(zip(incident_grid.axes, reference_values))

        # Get the reference grid point using the adjusted reference values
        incident_ref_grid_point = incident_grid.get_grid_point(**ref_dict)

        # Get the reference ionising photon luminosity
        reference_log10_specific_ionising_lum = (
            incident_grid.log10_specific_ionising_lum["HI"][
                incident_ref_grid_point])

    # reference_log10_specific_ionising_lum is used an an argument to
    # create_cloudy_input so it's necessary to set it to something here.
    else:
        reference_log10_specific_ionising_lum = None

    for incident_index, (incident_params_tuple, incident_index_tuple) in (
        enumerate(zip(incident_model_list, incident_index_list))):

        # get dictionary of incident parameters
        incident_parameters = dict(zip(incident_axes, incident_params_tuple))

        # print(incident_parameters)

        # Create a directory for each incident grid point
        Path(f"{output_directory}/{incident_index}").mkdir(
            parents=True,
            exist_ok=True)

        # Extract incident SED from the grid.
        lnu = incident_grid.spectra['incident'][tuple(incident_index_tuple)]

        # Save incident SED as a numpy array
        np.save(
            f"{output_directory}/{incident_index}/input",
            np.array([lam, lnu]))

        # Convert incided SED to cloudy format and save
        shape_commands = cloudy23.ShapeCommands.table_sed(
            'input',
            lam,
            lnu,
            output_dir=f"{output_directory}/{incident_index}")

        # Copy linelist to this folder
        shutil.copyfile(
            'linelist.dat',
            f'{output_directory}/{incident_index}/linelist.dat')

        # If the reference ionionisation parameter model is used we want
        # to calculate the difference between the specific ionising photon
        # luminosity for this grid point and the reference grid point
        if fixed_photoionisation_params["ionisation_parameter_model"] == "ref":
            delta_log10_specific_ionising_luminosity = (
                incident_grid.log10_specific_ionising_lum["HI"][
                    tuple(incident_index_tuple)]
                - reference_log10_specific_ionising_lum)
        # We still need to calculate delta_log10_specific_ionising_luminosity
        # since it is passed to the main function.
        else:
            delta_log10_specific_ionising_luminosity = None

        # Create cloudy input file for every photoionisation grid point

        # If we have photoionisation parameters that vary we need to loop over
        # all of the models
        if len(variable_photoionisation_params) > 0:

            for photoionisation_index, (
                photoionisation_params_tuple, photoionisation_index_tuple
            ) in enumerate(
                zip(photoionisation_model_list, photoionisation_index_list)
            ):

                # Get a dictionary of the photoionisation parameters that are
                # varying for this grid point
                variable_photoionisation_parameters = dict(
                    zip(photoionisation_axes, photoionisation_params_tuple))

                # Create dictionary of all photoionisation parameters
                photoionisation_parameters = (
                    fixed_photoionisation_params |
                    variable_photoionisation_parameters)

                # Create dictionary of all parameters
                parameters = incident_parameters | photoionisation_parameters

                # Create cloudy input model
                create_cloudy_input(
                    incident_index,
                    photoionisation_index,
                    parameters,
                    delta_log10_specific_ionising_luminosity,
                    output_directory,
                    )

        # Else, if there is only one photoionisation model there is no need to
        # loop.
        else:
            photoionisation_index = 0
            parameters = incident_parameters | fixed_photoionisation_params

            create_cloudy_input(
                incident_index,
                photoionisation_index,
                parameters,
                delta_log10_specific_ionising_luminosity,
                output_directory,
                )

    # If a specific machine is specified run the function in
    # submission_scripts to generate a submission script.
    if machine:
        getattr(submission_scripts, machine)(
            new_grid_name,
            number_of_incident_grid_points=incident_n_models,
            number_of_photoionisation_models=photoionisation_n_models,
            cloudy_output_dir=cloudy_output_dir,
            cloudy_executable_path=cloudy_executable_path,
            memory="4G",
            by_photoionisation_grid_point=by_photoionisation_grid_point)
