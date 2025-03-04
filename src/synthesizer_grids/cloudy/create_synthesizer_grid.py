"""A module for reading in cloudy outputs and creating a synthesizer grid.

This module reads in cloudy outputs and creates a synthesizer grid. The
synthesizer grid is a HDF5 file that contains the spectra and line luminosities
from cloudy outputs.

Example usage:
    python create_synthesizer_grid.py --grid-dir /path/to/grid
        --cloudy-dir /path/to/cloudy --incident-grid incident_grid.hdf5
        --cloudy-params c17.03-sps --include-spectra True --replace
        --line-calc-method linelist
"""

import os
import shutil

import h5py
import numpy as np
import yaml
from synthesizer.grid import Grid
from synthesizer.photoionisation import cloudy17, cloudy23
from synthesizer.sed import Sed
from synthesizer.units import has_units
from unyt import Angstrom, Hz, cm, dimensionless, erg, eV, s, yr
from utils import get_grid_props_cloudy

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser

# Create dictionary of commonly used units for grid axes
axes_units = {
    "reference_ionisation_parameter": dimensionless,
    "ionisation_parameter": dimensionless,
    "ages": yr,
    "metallicities": dimensionless,
    "hydrogen_density": cm ** (-3),
}


def create_empty_grid(
    grid_dir,
    incident_grid_name,
    new_grid_name,
    params,
    grid_params,
):
    """
    Create an empty Synthesizer grid file based on an existing incident grid.

    Args:
        grid_dir (str):
            Directory where the grid file will be saved.
        incident_grid_name (str):
            Name of the incident grid file we ran through cloudy.
        new_grid_name (str):
            Name of the new grid file.
        params (dict):
            Dictionary of parameters to be saved in the grid file.
        grid_params (dict):
            Dictionary of grid parameters.

    Returns:
        out_grid (GridFile):
            The newly created grid file.
        incident_grid (GridFile):
            The incident grid file.
    """
    # Open the parent incident grid
    incident_grid = Grid(
        incident_grid_name,
        grid_dir=grid_dir,
        read_lines=False,
    )

    # Make the new grid
    out_grid = GridFile(f"{grid_dir}/{new_grid_name}")

    # Add attribute with the original incident grid axes
    out_grid.write_attribute(
        group="/",
        attr_key="incident_axes",
        data=incident_grid.axes,
    )

    # Copy over the model metadata
    out_grid.write_model_metadata(incident_grid._model_metadata)

    # Set a list of the axes including the incident and new grid axes
    axes = list(incident_grid.axes) + list(grid_params.keys())

    # Add the incident grid parameters to grid_params
    for axis in incident_grid.axes:
        grid_params[axis] = getattr(incident_grid, axis)

    # Get properties of the grid
    (
        n_axes,
        shape,
        n_models,
        mesh,
        model_list,
        index_list,
    ) = get_grid_props_cloudy(axes, grid_params, verbose=True)

    # We want to copy over log10_specific_ionising_luminosity from the
    # incident grid to allow us to normalise the cloudy outputs.
    # However, the axes of the incident grid may be different from the
    # cloudy grid due to additional parameters, in which case we need to
    # extend the axes of log10_specific_ionising_luminosity.

    # Compute the amount we may have to expand the axes (if we don't need to
    # expand this is harmless)
    expansion = int(
        np.prod(shape)
        / np.prod(incident_grid.log10_specific_ionising_lum["HI"].shape)
    )

    # Loop over ions
    for ion in incident_grid.log10_specific_ionising_lum.keys():
        # If there are no additional axes simply copy over the incident
        # .log10_specific_ionising_lum .
        if len(axes) == len(incident_grid.axes):
            out_grid.write_dataset(
                key=f"log10_specific_ionising_luminosity/{ion}",
                data=incident_grid.log10_specific_ionising_lum[ion]
                * dimensionless,
                description="The specific ionising photon luminosity for"
                f" {ion}",
                log_on_read=False,
            )

        # Otherwise, we need to expand the axis to account for these additional
        # parameters.
        else:
            # Create new array with repeated elements
            log10_specific_ionising_luminosity = np.repeat(
                incident_grid.log10_specific_ionising_lum[ion] * dimensionless,
                expansion,
                axis=-1,
            )
            out_grid.write_dataset(
                f"log10_specific_ionising_luminosity/{ion}",
                np.reshape(log10_specific_ionising_luminosity, shape)
                * dimensionless,
                description="The specific ionising photon luminosity"
                f" for {ion}",
                log_on_read=False,
            )

    # Add attribute with the full grid axes
    out_grid.write_attribute(group="/", attr_key="axes", data=axes)

    # Copy over the weight attribute from the incident grid
    weight_var = getattr(incident_grid, "_weight_var", None)
    if weight_var is None:
        weight_var = "None"
    out_grid.write_attribute(
        group="/",
        attr_key="WeightVariable",
        data=weight_var,
    )

    # Now write out the grid axes, we first do the incident grid axes so we
    # can extract their metadata and then any extras
    for axis, log_on_read in zip(
        incident_grid.axes, incident_grid._logged_axes
    ):
        # Write out this axis (we can get everything we need from the incident)
        out_grid.write_dataset(
            key=f"axes/{axis}",
            data=getattr(incident_grid, axis),
            description=f"Grid axes {axis} (taken from incident grid)",
            log_on_read=log_on_read,
        )

    # Now write out the new grid axes if there are any
    for axis in axes:
        # Skip the one we did above
        if axis in incident_grid.axes:
            continue

        # If we have units in the grid_params dictionary already then use
        # these, otherwise use the axes_units dictionary, if we don't have
        # units for the axis then raise an error.
        if has_units(grid_params[axis]):
            out_grid.write_dataset(
                key=f"axes/{axis}",
                data=grid_params[axis],
                description=f"grid axes {axis} (introduced during cloudy run)",
                log_on_read=False,
            )

        # Ok, maybe its just a common axis that we have units for
        elif axis in axes_units:
            # Multiply values with the corresponding unit
            values_with_units = grid_params[axis] * axes_units[axis]

            out_grid.write_dataset(
                key=f"axes/{axis}",
                data=values_with_units,
                description=f"grid axes {axis} (introduced during cloudy run)",
                log_on_read=False,
            )

        else:
            raise ValueError(
                f"Unit for axis '{axis}' needs to be added to "
                f"create_synthesizer_grid.py"
            )

    # Write out cloudy parameters
    out_grid.write_cloudy_metadata(params)

    return out_grid, incident_grid


def load_grid_params(param_file="c23.01-sps"):
    """
    Read parameters from a yaml parameter file.

    This sorts the parameters into fixed and grid axis parameters. Fixed
    parameters are singular values that were fed into cloudy. Grid axis
    parameters are parameters with a range of values which form one of the
    axes of the eventual grid.

    Arguments:
        param_file (str)
            Path to the the parameter file

    Returns:
        fixed_params (dict)
            dictionary of parameters that are fixed
        grid_params (dict)
            dictionary of parameters that vary on the grid
    """
    # Open parameter file
    with open(param_file, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    grid_params = {}
    fixed_params = {}

    # Loop over parameters
    for k, v in params.items():
        # If parameter is a list store it in the grid_parameters dictionary
        # and convert to a numpy array
        if isinstance(v, list):
            grid_params[k] = np.array(list(map(float, v)))

        # Otherwise store it in fixed_params dictionary
        else:
            fixed_params[k] = v

    return fixed_params, grid_params


def get_grid_properties_hf(hf, verbose=True):
    """
    Get the properties of a grid from an HDF5 file.

    Arguments:
        hf : h5py.File
            The HDF5 file containing the grid.
        verbose : boolean
            Print out the properties of the grid.

    Returns:
        axes : list
            List of axes in the correct order.
        n_axes : int
            Number of axes in the grid.
        shape : tuple
            Shape of the grid.
        n_models : int
            Number of models in the grid.
        mesh : tuple
            Mesh of the grid.
        model_list : list
            List of model parameters.
        index_list : list
            List of model indices.
    """
    # Unpack the axes
    axes = hf.attrs["axes"]
    axes_values = {axis: hf[f"axes/{axis}"][:] for axis in axes}

    # Get the properties of the grid including the dimensions etc.
    return axes, *get_grid_props_cloudy(axes, axes_values, verbose=verbose)


def check_cloudy_runs(
    out_grid,
    cloudy_dir,
    replace=False,
    files_to_check=["cont"],
):
    """
    Check that all the cloudy runs have run properly.

    Args:
        out_grid (GridFile)
            The grid file to check.
        cloudy_dir (str)
            Parent directory for the cloudy runs.
        replace (boolean)
            If a run has failed simply replace the model with the previous one.
        files_to_check (list)
            List of files to check for each model.

    Returns:
        failed_list : list
            List of models that have failed.
    """
    # Get the properties of the grid
    axes = out_grid.read_attribute("axes")
    axes_values = {
        axis: out_grid.read_dataset(f"axes/{axis}") for axis in axes
    }
    (
        n_axes,
        shape,
        n_models,
        mesh,
        model_list,
        index_list,
    ) = get_grid_props_cloudy(axes, axes_values, verbose=True)

    # List of failed models
    failed_list = []
    for i, grid_params_ in enumerate(model_list):
        infile = f"{cloudy_dir}/{i+1}"
        failed = False

        # Check if files exist
        for ext in files_to_check:
            if not os.path.isfile(infile + "." + ext):
                failed = True

        # If they exist also check they have size >0
        if not failed:
            for ext in files_to_check:
                if os.path.getsize(infile + "." + ext) < 1000:
                    failed = True

        if failed:
            print(i + 1, model_list[i])
            failed_list.append(i + 1)

            """
            If replace is specified, instead replace the grid point with
            the previous one. NOTE: this should be a last resort if the
            cloudy runs of a small number of grid points are consistently
            failing.
            """
            if replace:
                for ext in files_to_check:
                    shutil.copyfile(
                        f"{cloudy_dir}/{i}.{ext}",
                        infile + f".{ext}",
                    )

    # If the files have been replace set the failed list to empty so the
    # rest of the code can run.
    if replace:
        failed_list = []
    return failed_list


def add_spectra(
    out_grid,
    incident_grid,
    grid_name,
    cloudy_dir,
    spec_names=("incident", "transmitted", "nebular", "linecont"),
    weight="initial_masses",
    norm_by_q=True,
):
    """
    Open cloudy spectra and add them to the grid.

    Args:
        out_grid (GridFile)
            The grid file to add the spectra to.
        incident_grid (GridFile)
            The incident grid.
        cloudy_dir (str)
            Parent directory for the cloudy runs.
        spec_names (list)
            The names of the spectra to save (default is incident, transmitted,
            nebular and linecont).
        weight (str)
            The weight variable by which the spectra have been normalised. For
            example, most SPS models are per initial stellar mass, so the
            weight variable would be 'initial_masses', the Synthesizer
            variable.
        norm_by_q (bool)
            If True, the spectra are normalised by the specific ionising
            photon luminosity calculated from the incident spectrum.
    """
    # Get the properties of the grid
    axes = out_grid.read_attribute("axes")
    axes_values = {
        axis: out_grid.read_dataset(f"axes/{axis}") for axis in axes
    }
    (
        n_axes,
        shape,
        n_models,
        mesh,
        model_list,
        index_list,
    ) = get_grid_props_cloudy(axes, axes_values, verbose=True)

    # Determine the cloudy version...
    cloudy_version = out_grid.read_attribute(
        "cloudy_version",
        group="CloudyParams",
    )

    # ... and use to select the correct module.
    if cloudy_version.split(".")[0] == "c23":
        cloudy = cloudy23
    elif cloudy_version.split(".")[0] == "c17":
        cloudy = cloudy17

    # Read first spectra from the first grid point to get length and
    # wavelength grid.
    lam = cloudy.read_wavelength(f"{cloudy_dir}/1")

    # Write the spectra names... for some reason (not sure why since .keys()
    # gives the same result)
    out_grid.write_attribute(group="/", attr_key="spec_names", data=spec_names)

    # Number of wavelength points
    nlam = len(lam)

    # Set up the spectra dictionary
    spectra = {}

    # Make spectral grids and set them to zero
    for spec_name in spec_names:
        spectra[spec_name] = np.zeros((*shape, nlam))

    # Array for holding the normalisation which is calculated below and
    # used by lines
    spectra["normalisation"] = np.ones(shape)

    # Loop over the grid points
    for i, indices in enumerate(index_list):
        indices = tuple(indices)

        # Define the infile (cloudy output file)
        infile = f"{cloudy_dir}/{i+1}"

        # Read the continuum file containing the spectra
        spec_dict = cloudy.read_continuum(infile, return_dict=True)

        # Calculate the specific ionising photon luminosity and use this to
        # renormalise the spectrum.
        if norm_by_q:
            # Create sed object
            sed = Sed(
                lam=lam * Angstrom,
                lnu=spec_dict["incident"] * erg / s / Hz,
            )

            # Calculate Q
            ionising_photon_production_rate = (
                sed.calculate_ionising_photon_production_rate(
                    ionisation_energy=13.6 * eV, limit=100
                )
            )

            # Calculate normalisation
            normalisation = out_grid.read_dataset(
                "log10_specific_ionising_luminosity/HI",
                indices=indices,
            ) - np.log10(ionising_photon_production_rate)

            # Save normalisation for later use (rescaling lines)
            spectra["normalisation"][indices] = 10**normalisation

        # Save the normalised spectrum to the correct grid point
        for spec_name in spec_names:
            spectra[spec_name][indices] = (
                spec_dict[spec_name] * spectra["normalisation"][indices]
            )

    # Apply units to the spectra
    for spec in spectra:
        spectra[spec] *= erg / s / Hz

    # Write the spectra out
    weight_var = getattr(incident_grid, "_weight_var", None)
    if weight_var is None:
        weight_var = "None"
    out_grid.write_spectra(
        spectra,
        wavelength=lam * Angstrom,
        weight=weight_var,  # this is written already but harmless
    )

    return spectra


def add_lines(
    grid_name,
    grid_dir,
    cloudy_dir,
    line_type="linelist",
    lines_to_include=False,
    include_spectra=True,
):
    """
    Open cloudy lines and add them to the HDF5 grid

    Arguments:
        grid_name : str
            Name of the grid
        grid_dir : str
            Parent directory for the grids.
        cloudy_dir : str
            Parent directory for the cloudy runs.
        line_type : str
            The type of line file to use (linelist, lines)
        dlog10_specific_ionising_lum
            The difference between the original and cloudy
            log10_specific_ionising_lum used for rescaling the cloudy spectra
    """

    # open the new grid
    with h5py.File(f"{grid_dir}/{grid_name}", "a") as hf:
        # Get the properties of the grid including the dimensions etc.
        (
            axes,
            n_axes,
            shape,
            n_models,
            mesh,
            model_list,
            index_list,
        ) = get_grid_properties_hf(hf)

        # Determine the cloudy version...
        cloudy_version = out_grid.read_attribute(
            "cloudy_version",
            group="CloudyParams",
        )

        # ... and use to select the correct module.
        if cloudy_version.split(".")[0] == "c23":
            cloudy = cloudy23
        elif cloudy_version.split(".")[0] == "c17":
            cloudy = cloudy17

        # delete lines group if it already exists
        if "lines" in hf:
            del hf["lines"]

        # define spectra
        if include_spectra:
            spectra = hf["spectra"]
            normalisation = spectra["normalisation"][:]
            lam = spectra["wavelength"][:]

        # create group for holding lines
        lines = hf.create_group("lines")

        if line_type == "linelist":
            infile = f"{cloudy_dir}/1"
            lines_to_include, _, _ = cloudy.read_linelist(
                infile, extension="emergent_elin"
            )

        # set up output arrays
        for line_id in lines_to_include:
            lines[f"{line_id}/luminosity"] = np.zeros(shape)
            lines[f"{line_id}/transmitted_continuum"] = np.zeros(shape)
            lines[f"{line_id}/nebular_continuum"] = np.zeros(shape)
            lines[f"{line_id}/continuum"] = np.zeros(shape)

        for i, indices in enumerate(index_list):
            # convert indices array to tuple
            indices = tuple(indices)

            # define the infile
            infile = f"{cloudy_dir}/{i+1}"

            # get TOTAL continuum spectra
            if include_spectra:
                nebular_continuum = (
                    spectra["nebular"][indices] - spectra["linecont"][indices]
                )
                continuum = spectra["transmitted"][indices] + nebular_continuum

            # get line quantities  <--- THIS NEEDS TO CHANGE

            if line_type == "lines":
                (
                    id,
                    blend,
                    wavelength,
                    intrinsic,
                    luminosity,
                ) = cloudy.read_lines(infile)

                # identify lines we want to keep
                s = np.nonzero(np.in1d(id, np.array(lines_to_include)))[0]

                id = id[s]
                wavelength = wavelength[s]
                luminosity = luminosity[s]

            elif line_type == "linelist":
                id, wavelength, luminosity = cloudy.read_linelist(
                    infile, extension="emergent_elin"
                )

            else:
                raise ValueError(
                    f"Unrecognised line type: {line_type} "
                    "(must be 'lines' or 'linelist')"
                )

            for id_, wavelength_, luminosity_ in zip(
                id, wavelength, luminosity
            ):
                line = lines[id_]

                # save line wavelength
                line.attrs["wavelength"] = wavelength_

                # if spectra have been calculated pull the normalisation
                if include_spectra:
                    norm = normalisation[indices]
                else:
                    norm = 1.0

                # Calculate line luminosity and save it. Uses normalisation
                # from spectra.
                line["luminosity"][indices] = luminosity_ * norm  # erg s^-1

                if include_spectra:
                    # Calculate transmitted continuum at the line wavelength
                    # and save it.
                    line["transmitted_continuum"][indices] = np.interp(
                        wavelength_, lam, spectra["transmitted"][indices]
                    )  # erg s^-1 Hz^-1

                    # calculate nebular continuum at the line wavelength and
                    # save it.
                    line["nebular_continuum"][indices] = np.interp(
                        wavelength_, lam, nebular_continuum
                    )  # erg s^-1 Hz^-1

                    # calculate total continuum at the line wavelength and
                    # save it.
                    line["continuum"][indices] = np.interp(
                        wavelength_, lam, continuum
                    )  # erg s^-1 Hz^-1


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(
        description="Create Synthesizer HDF5 grid files from cloudy outputs.",
        cloudy_args=True,
    )
    args = parser.parse_args()

    # Unpack arguments
    grid_dir = args.grid_dir
    cloudy_dir = args.cloudy_dir
    incident_grid_name = args.incident_grid
    new_grid_name = (
        args.cloudy_grid
        if args.cloudy_grid is not None
        else incident_grid_name
    )
    cloudy_paramfile = args.cloudy_params
    extra_cloudy_paramfile = args.cloudy_params_addition
    include_spectra = args.include_spectra
    replace_failed = args.replace
    line_type = args.line_calc_method
    machine = args.machine
    verbose = args.verbose

    # Check for extensions
    if cloudy_paramfile.split(".")[-1] != "yaml":
        cloudy_paramfile += ".yaml"
    if extra_cloudy_paramfile is not None:
        if extra_cloudy_paramfile.split(".")[-1] != "yaml":
            extra_cloudy_paramfile += ".yaml"
    if incident_grid_name.split(".")[-1] != "hdf5":
        incident_grid_name += ".hdf5"
    if new_grid_name.split(".")[-1] != "hdf5":
        new_grid_name += ".hdf5"

    # Define the grid name (the user has either set this or its the same as
    # the incident grid name with a cloudy suffix)
    if new_grid_name is None:
        new_grid_name = incident_grid_name
    if "cloudy" not in new_grid_name:
        new_grid_name = f"{new_grid_name.replace('.hdf5', '')}_cloudy.hdf5"

    # Load the cloudy parameters you are going to run
    fixed_params, grid_params = load_grid_params(args.cloudy_params)

    # If an additional parameter set is provided supersede the default
    # parameters with these and adjust the new grid name.
    if args.cloudy_params_addition:
        additional_fixed_params, additional_grid_params = load_grid_params(
            args.cloudy_params_addition
        )
        fixed_params = fixed_params | additional_fixed_params
        grid_params = grid_params | additional_grid_params

    params = fixed_params | grid_params

    print(params)

    # Create empty synthesizer grid, this returns a GridFile object we can
    # add spectra and lines to below
    out_grid, incident_grid = create_empty_grid(
        grid_dir,
        incident_grid_name,
        new_grid_name,
        params,
        grid_params,
    )

    # Check cloudy runs and potentially replace them by the nearest grid point
    # if they fail.
    failed_list = check_cloudy_runs(
        out_grid,
        cloudy_dir,
        replace=args.replace,
    )
    print("list of failed cloudy runs:", failed_list)

    # If any runs have failed prompt to re-run.
    if len(failed_list) > 0:
        # get number of failed runs
        n_fail = len(failed_list)

        print(f"ERROR: {n_fail} cloudy runs have failed.")

        if args.machine == "apollo":
            # apollo specific command
            print("Re-run with this command:")
            print(f"qsub -t 1:{n_fail} run_grid.job")

            # replace input_names with list of failed runs
            with open(
                f"{cloudy_dir}/{incident_grid_name}/input_names.txt", "w"
            ) as myfile:
                myfile.write("\n".join(map(str, failed_list)))

    # If no runs have failed, go ahead and add spectra and lines.
    else:
        # Add spectra
        if include_spectra:
            add_spectra(
                out_grid,
                incident_grid,
                new_grid_name,
                cloudy_dir,
                spec_names=("incident", "transmitted", "nebular", "linecont"),
                weight="initial_masses",
                norm_by_q=True,
            )

        # add lines
        add_lines(
            new_grid_name,
            grid_dir,
            cloudy_dir,
            line_type="linelist",
            include_spectra=include_spectra,
        )
