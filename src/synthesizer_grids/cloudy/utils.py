import numpy as np
import yaml


def get_cloudy_params(
        param_file="c23.01-test",
        param_dir="params"):
    """
    Read parameters from a yaml parameter file

    Arguments:
        param_file (str)
            filename of the parameter file
        param_dir (str)
            directory containing the parameter file

    Returns:
        fixed_params (dict)
            dictionary of parameters that are fixed
        grid_params (dict)
            dictionary of parameters that vary on the grid
    """

    # open paramter file
    with open(f"{param_dir}/{param_file}", "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    grid_params = {}
    fixed_params = {}

    # loop over parameters
    for k, v in params.items():

        # if parameter is a list store it in the grid_parameters dictionary
        # and convert to a numpy array
        if isinstance(v, list):
            grid_params[k] = np.array(list(map(float, v)))

        # if a dictionary collect any parameters that are also lists
        elif isinstance(v, dict):
            for k_, v_ in v.items():
                if isinstance(v_, list):
                    grid_params[f'{k}.{k_}'] = np.array(list(map(float, v_)))
                else:
                    fixed_params[f'{k}.{k_}'] = v_

        # otherwise store it in fixed_params dictionary
        else:
            fixed_params[k] = v

    return fixed_params, grid_params


def get_grid_props_cloudy(axes, axes_values, verbose=True):
    """
    Get the cloudy related properties of a grid.

    This returns some basic properties needed for processing an incident
    grid through CLOUDY.

    Args:
        axes (list)
            List of axes to vary.
        axes_values (dict)
            Dictionary of axis values.
        verbose (bool)
            Are we talking or not?

    Returns:
        n_axes (int)
            Number of axes.
        shape (list)
            Shape of the grid.
        n_models (int)
            Number of models.
        mesh (numpy.ndarray)
            Mesh of the grid.
        model_list (numpy.ndarray)
            List of models.
        index_list (numpy.ndarray)
            List of indices of each model within the grid.
    """
    # The grid axes
    if verbose:
        print(f"axes: {axes}")

    # Number of axes
    n_axes = len(axes)
    if verbose:
        print(f"number of axes: {n_axes}")

    # The shape of the grid (useful for creating outputs)
    shape = list([len(axes_values[axis]) for axis in axes])
    if verbose:
        print(f"shape: {shape}")

    # Determine number of models
    n_models = np.prod(shape)
    if verbose:
        print(f"number of models to run: {n_models}")

    # Create the mesh of the grid
    mesh = np.array(
        np.meshgrid(*[np.array(axes_values[axis]) for axis in axes])
    )

    # Create the list of the models
    model_list = mesh.T.reshape(n_models, n_axes)
    if verbose:
        print("model list:")
        print(model_list)

    # Create a list of the indices
    index_mesh = np.array(np.meshgrid(*[range(n) for n in shape]))
    index_list = index_mesh.T.reshape(n_models, n_axes)
    if verbose:
        print("index list:")
        print(index_list)

    return n_axes, shape, n_models, mesh, model_list, index_list

