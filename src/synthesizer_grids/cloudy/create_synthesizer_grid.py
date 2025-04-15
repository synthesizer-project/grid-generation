"""A module for reading in cloudy outputs and creating a synthesizer grid.

This module reads in cloudy outputs and creates a synthesizer grid. The
synthesizer grid is a HDF5 file that contains the spectra and line luminosities
from cloudy outputs.

Example usage:

"""

import os
import re

import numpy as np
from synthesizer.emissions import Sed
from synthesizer.grid import Grid
from synthesizer.photoionisation import cloudy17, cloudy23
from synthesizer.units import has_units
from unyt import Angstrom, Hz, cm, dimensionless, erg, eV, km, s, yr
from utils import (
    get_cloudy_params,
    get_grid_props_cloudy,
)

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser

# Create dictionary of commonly used units for grid axes
axes_units = {
    "reference_ionisation_parameter": dimensionless,
    "fixed_ionisation_parameter": dimensionless,
    "depletion_scale": dimensionless,
    "z": dimensionless,
    "ionisation_parameter": dimensionless,
    "ages": yr,
    "metallicities": dimensionless,
    "hydrogen_density": cm ** (-3),
    "stop_column_density": dimensionless,
    "alpha_enhancement": dimensionless,
    "abundance_scalings.nitrogen_to_oxygen": dimensionless,
    "abundance_scalings.carbon_to_oxygen": dimensionless,
    "turbulence": km / s,
}

log_on_read_dict = {
    "reference_ionisation_parameter": True,
    "fixed_ionisation_parameter": True,
    "depletion_scale": False,
    "z": False,
    "ionisation_parameter": True,
    "hydrogen_density": True,
    "stop_column_density": False,
    "alpha_enhancement": False,
    "abundance_scalings.nitrogen_to_oxygen": False,
    "abundance_scalings.carbon_to_oxygen": False,
}

pluralisation_of_axes = {
    "z": "redshifts",
    "metallicities": "metallicities",
    "ionisation_parameter": "ionisation_parameters",
    "turbulence": "turbulences",
    "hydrogen_density": "hydrogen_densities",
    "depletion_scale": "depletion_scales",
    "stop_column_density": "column_densities",
    "alpha_enhancement": "alpha_enhancements",
    "reference_ionisation_parameter": "reference_ionisation_parameters",
}


def create_empty_grid(
    grid_dir,
    incident_grid_name,
    new_grid_name,
    new_axes_values,
    photoionisation_parameters,
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
        new_axes_values (dict):
            Dictionary of the new axes values.
        photoionisation_params (dict):
            Dictionary of photoionisation modelling parameters.

    Returns:
        out_grid (GridFile):
            The newly created grid file.
        incident_grid (GridFile):
            The incident grid file.
        shape (tuple)
            The shape of the new grid.
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
    new_axes = list(new_axes_values.keys())

    # Get properties of the grid
    (
        n_axes,
        shape,
        n_models,
        mesh,
        model_list,
        index_list,
    ) = get_grid_props_cloudy(new_axes, new_axes_values, verbose=False)

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
        if len(new_axes) == len(incident_grid.axes):
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

    # Copy over the weight attribute from the incident grid
    weight_var = getattr(incident_grid, "_weight_var", None)
    if weight_var is None:
        weight_var = "None"
    out_grid.write_attribute(
        group="/",
        attr_key="WeightVariable",
        data=weight_var,
    )

    print(incident_grid._extract_axes)

    # Now write out the grid axes, we first do the incident grid axes so we
    # can extract their metadata and then any extras
    for axis, extract_axis in zip(
        incident_grid.axes, incident_grid._extract_axes
    ):
        # Write out this axis (we can get everything we need from the incident)
        if extract_axis != axis:
            log_on_read = True
        else:
            log_on_read = False

        out_grid.write_dataset(
            key=f"axes/{axis}",
            data=getattr(incident_grid, axis),
            description=f"Grid axes {axis} (taken from incident grid)",
            log_on_read=log_on_read,
        )

    # Now write out the new grid axes if there are any
    # List of new grid axes including pluralised form
    new_axes_for_grid = []
    for axis in new_axes:
        # Skip the one we did above
        if axis in incident_grid.axes:
            new_axes_for_grid.append(axis)
            continue

        if axis in pluralisation_of_axes.keys():
            pluralised_axis = pluralisation_of_axes[axis]
            new_axes_for_grid.append(pluralised_axis)
        else:
            raise ValueError(f"No pluralisation set for {axis} axis")

        # If we have units in the grid_params dictionary already then use
        # these, otherwise use the axes_units dictionary, if we don't have
        # units for the axis then raise an error.
        if has_units(new_axes_values[axis]):
            out_grid.write_dataset(
                key=f"axes/{pluralised_axis}",
                data=new_axes_values[axis],
                description=f"grid axes {pluralised_axis} (introduced during"
                "cloudy run)",
                log_on_read=log_on_read_dict[axis],
            )

        # Ok, maybe its just a common axis that we have units for
        elif axis in axes_units:
            # Multiply values with the corresponding unit
            values_with_units = new_axes_values[axis] * axes_units[axis]

            out_grid.write_dataset(
                key=f"axes/{pluralised_axis}",
                data=values_with_units,
                description=f"grid axes {axis} (introduced during cloudy run)",
                log_on_read=False,
            )

        else:
            raise ValueError(
                f"Unit for axis '{axis}' needs to be added to "
                f"create_synthesizer_grid.py"
            )

    # Add attribute with the full grid axes
    out_grid.write_attribute(
        group="/", attr_key="axes", data=new_axes_for_grid
    )

    # Write out cloudy parameters
    out_grid.write_cloudy_metadata(photoionisation_parameters)

    return out_grid, incident_grid


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
    new_grid_name,
    cloudy_dir,
    incident_index_list,
    photoionisation_index_list,
    files_to_check=["emergent_elin"],
):
    """
    Check that all the cloudy runs have run properly.

    Args:
        new_grid_name
            The name of the new grid (also the subdirectory wher the cloudy
            runs are stored)
        cloudy_dir (str)
            Parent directory for the cloudy runs.
        incident_index_list (list)
            List of incident grid points
        photoionisation_index_list (list)
            List of photoionisation grid points
        replace (boolean)
            If a run has failed simply replace the model with the previous one.
        files_to_check (list)
            List of files to check for each model.

    Returns:
        failed_list : list
            List of models that have failed.
    """

    # List of tuples describing failed models
    failed_list = []

    for incident_index, incident_index_tuple in enumerate(incident_index_list):
        for photoionisation_index, photoionisation_index_tuple in enumerate(
            photoionisation_index_list
        ):
            # The infile
            infile = (
                f"{cloudy_dir}/{new_grid_name}/"
                f"{incident_index}/{photoionisation_index}"
            )

            failed = False

            # Check if files exist
            for ext in files_to_check:
                if not os.path.isfile(infile + "." + ext):
                    failed = True

            # If they exist also check they have size >0
            if not failed:
                for ext in files_to_check:
                    if os.path.getsize(infile + "." + ext) < 100:
                        failed = True

            # Record models that have failed
            if failed:
                print(incident_index, photoionisation_index)
                failed_list.append((incident_index, photoionisation_index))

    return failed_list


def add_spectra(
    new_grid,
    new_grid_name,
    cloudy_dir,
    incident_index_list,
    photoionisation_index_list,
    new_shape,
    spec_names=("incident", "transmitted", "nebular", "linecont"),
    norm_by_q=True,
):
    """
    Open cloudy spectra and add them to the grid.

    Args:
        new_grid (GridFile)
            The grid file to add the spectra to.
        new_grid_name (str)
            The name of the new grid.
        cloudy_dir (str)
            Parent directory for the cloudy runs.
        incident_index_list (list)
            List of incident grid points
        photoionisation_index_list (list)
            List of photoionisation grid points
        spec_names (list)
            The names of the spectra to save (default is incident, transmitted,
            nebular and linecont).
        shape (ndarray)
            The shape of the new grid.
        norm_by_q (bool)
            If True, the spectra are normalised by the specific ionising
            photon luminosity calculated from the incident spectrum.
    """

    # Determine the cloudy version...
    cloudy_version = new_grid.read_attribute(
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
    lam = cloudy.read_wavelength(f"{cloudy_dir}/{new_grid_name}/0/0")

    # Write the spectra names... for some reason (not sure why since .keys()
    # gives the same result)
    new_grid.write_attribute(group="/", attr_key="spec_names", data=spec_names)

    # Number of wavelength points
    nlam = len(lam)

    # Set up the spectra dictionary
    spectra = {}

    # Make spectral grids and set them to zero
    for spec_name in spec_names:
        spectra[spec_name] = np.zeros((*new_shape, nlam))

    # Array for holding the normalisation which is calculated below and
    # used by lines
    spectra["normalisation"] = np.ones(new_shape)

    for incident_index, incident_index_tuple in enumerate(incident_index_list):
        for photoionisation_index, photoionisation_index_tuple in enumerate(
            photoionisation_index_list
        ):
            # Get the indices of the grid_point
            indices = tuple(incident_index_tuple) + tuple(
                photoionisation_index_tuple
            )

            # The infile
            infile = (
                f"{cloudy_dir}/{new_grid_name}/"
                f"{incident_index}/{photoionisation_index}"
            )

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
                normalisation = new_grid.read_dataset(
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
    new_grid.write_spectra(
        spectra,
        wavelength=lam * Angstrom,
    )

    return lam, spectra


def add_lines(
    new_grid,
    new_grid_name,
    cloudy_dir,
    incident_index_list,
    photoionisation_index_list,
    new_shape,
    lam=None,
    spectra=None,
    line_type="linelist",
):
    """
    Open cloudy lines and add them to the HDF5 grid

    Arguments:
        new_grid (GridFile)
            The grid file to add the spectra to.
        new_grid_name (str)
            The name of the new grid.
        cloudy_dir (str)
            Parent directory for the cloudy runs.
        incident_index_list (list)
            List of incident grid points
        photoionisation_index_list (list)
            List of photoionisation grid points
        new_shape (ndarray)
            The shape of the new grid.
        lam (ndarray)
            The wavelength grid of the spectra.
        spectra (bool)
            The spectra.
        line_type (str)
            The type of line file to use (linelist, lines)
    """

    if spectra is not None:
        calculate_continuum = True
    else:
        calculate_continuum = False

    # Determine the cloudy version...
    cloudy_version = new_grid.read_attribute(
        "cloudy_version",
        group="CloudyParams",
    )

    # ... and use to select the correct module.
    if cloudy_version.split(".")[0] == "c23":
        cloudy = cloudy23
    elif cloudy_version.split(".")[0] == "c17":
        cloudy = cloudy17

    # Open the first linelist and use it to populate the keys
    with open(f"{cloudy_dir}/{new_grid_name}/0/linelist.dat") as file:
        linelist = file.readlines()
        # strip escape character
        lines_to_include = [
            re.sub(r"[\x00-\x1F\x7F]", "", s) for s in linelist
        ]

    # Calculate number of lines
    nlines = len(lines_to_include)

    print(f"number of lines included: {nlines}")

    # Set up the lines dictionary
    lines = {}

    # Define output quantities

    # We always save luminosity...
    lines["luminosity"] = np.empty((*new_shape, nlines))

    # ... but only save continuum values if spectra are provided.
    if calculate_continuum:
        continuum_quantities = [
            "transmitted",
            "nebular_continuum",
            "total_continuum",
        ]

        spectra_ = {}

        # Calculate transmitted_contiuum
        spectra_["transmitted"] = spectra["transmitted"]

        # Calculate nebular_contiuum
        spectra_["nebular_continuum"] = (
            spectra["nebular"] - spectra["linecont"]
        )

        # Calculate total_continuum. Note this is not the same as adding
        #  nebular and transmitted since that would include emission lines.
        spectra_["total_continuum"] = (
            spectra_["nebular_continuum"] + spectra_["transmitted"]
        )

        # Define output arrays
        for continuum_quantity in continuum_quantities:
            lines[continuum_quantity] = np.empty((*new_shape, nlines))

    # Loop over incident models and photoionisation models
    for incident_index, incident_index_tuple in enumerate(incident_index_list):
        for photoionisation_index, photoionisation_index_tuple in enumerate(
            photoionisation_index_list
        ):
            # Get the indices of the grid_point
            indices = tuple(incident_index_tuple) + tuple(
                photoionisation_index_tuple
            )

            # The infile
            infile = (
                f"{cloudy_dir}/{new_grid_name}/"
                f"{incident_index}/{photoionisation_index}"
            )

            # Read line quantities
            ids, line_wavelengths, luminosities = cloudy.read_linelist(
                infile, extension="emergent_elin"
            )

            # re-order by wavelength
            sorted_indices = np.argsort(line_wavelengths)
            ids = ids[sorted_indices]
            luminosities = luminosities[sorted_indices]
            line_wavelengths = line_wavelengths[sorted_indices]

            # If we're on the first grid point save the wavelength grid
            if (incident_index == 0) and (photoionisation_index == 0):
                lines["wavelength"] = line_wavelengths
                # for id, lam in zip(ids, sorted_indices):
                #     print(id, lam)

                # Record list of IDs
                lines["id"] = ids

            # If spectra have been calculated extract the normalisation ..
            if calculate_continuum:
                normalisation = spectra["normalisation"][indices]

            # Otherwise set the normalisation to unity
            else:
                normalisation = 1.0

            # Calculate line luminosity and save it. Uses normalisation
            # from spectra. Units are erg/s
            lines["luminosity"][indices] = luminosities * normalisation

            # Calculate continuum luminosities. Units are erg/s/Hz.
            if calculate_continuum:
                for continuum_quantity in continuum_quantities:
                    lines[continuum_quantity][indices] = np.interp(
                        line_wavelengths,
                        lam,
                        spectra_[continuum_quantity][indices],
                    )

    # Apply units to the wavelength
    lines["wavelength"] *= Angstrom

    # Apply units to the lines
    lines["luminosity"] *= erg / s

    # If continuum values are calculated add units
    if calculate_continuum:
        for continuum_quantity in continuum_quantities:
            lines[continuum_quantity] *= erg / s / Hz

    # Write the lines out
    new_grid.write_lines(
        lines,
    )


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(
        description="Create Synthesizer HDF5 grid files from cloudy outputs.",
        cloudy_args=True,
    )

    # Add additional parameters which are specific to this script

    # Should we include the spectra in the grid?
    parser.add_argument(
        "--include-spectra",
        action="store_true",
        help="Should the spectra be included in the grid?",
        default=False,
        required=False,
    )

    # Should we normalise by the specific ionising luminosity?
    parser.add_argument(
        "--norm-by-Q",
        "-Q",
        default=True,
        required=False,
        help="Should the grid be normalised by the specific "
        "ionising luminosity? (default is True)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Unpack arguments
    grid_dir = args.grid_dir
    cloudy_dir = args.cloudy_output_dir
    incident_grid_name = args.incident_grid
    cloudy_param_file = args.cloudy_paramfile
    extra_cloudy_param_file = args.cloudy_paramfile_extra
    include_spectra = args.include_spectra
    # line_type = args.line_calc_method
    verbose = args.verbose

    print(" " * 80)
    print("-" * 80)
    print(incident_grid_name)
    print(cloudy_param_file)
    print(extra_cloudy_param_file)

    # Check for extensions
    # Create _file and _name versions (with and w/o extensions)

    if cloudy_param_file.split(".")[-1] != "yaml":
        cloudy_param_name = cloudy_param_file
        cloudy_param_file += ".yaml"
    else:
        cloudy_param_name = "".join(cloudy_param_file.split(".")[:-1])

    if extra_cloudy_param_file is not None:
        if extra_cloudy_param_file.split(".")[-1] != "yaml":
            extra_cloudy_param_name = extra_cloudy_param_file
            extra_cloudy_param_file += ".yaml"
        else:
            extra_cloudy_param_name = "".join(
                extra_cloudy_param_file.split(".")[:-1]
            )

    if incident_grid_name.split(".")[-1] != "hdf5":
        incident_grid_file = incident_grid_name + ".hdf5"
    else:
        incident_grid_file = incident_grid_name
        incident_grid_name = "".join(incident_grid_name.split(".")[:-1])

    # Get name of new grid (concatenation of incident_grid and cloudy
    # parameter file)
    new_grid_name = f"{incident_grid_name}_cloudy-{cloudy_param_name}"

    # If an additional parameter set append this to the new grid name
    if extra_cloudy_param_file:
        # Ignore the directory part if it exists
        extra_cloudy_param_name = extra_cloudy_param_name.split("/")[-1]
        # Append the new_grid_name with the additional name
        new_grid_name += "-" + extra_cloudy_param_name

    new_grid_file = new_grid_name + ".hdf5"
    print(cloudy_param_name, cloudy_param_file)
    print(new_grid_name, new_grid_file)

    # Open the incident grid using synthesizer
    incident_grid = Grid(
        incident_grid_name,
        grid_dir=grid_dir,
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
        incident_axes, incident_axes_values, verbose=False
    )

    # Load the cloudy parameters you are going to run
    fixed_photoionisation_params, variable_photoionisation_params = (
        get_cloudy_params(cloudy_param_file)
    )

    # If an additional parameter set is provided supersede the default
    # parameters with these.
    if extra_cloudy_param_file:
        (photoionisation_fixed_params_, photoionisation_variable_params_) = (
            get_cloudy_params(extra_cloudy_param_file)
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
            verbose=False,
        )

    # Else, we need to specify that the only photonionisation model is 0
    else:
        photoionisation_n_models = 1
        photoionisation_index_list = [[]]
        photoionisation_shape = ()

    # Define the axes and values of the output grid
    new_axes = incident_axes
    new_axes_values = incident_axes_values

    new_shape = tuple(incident_shape) + tuple(photoionisation_shape)
    if verbose:
        print("new shape:", new_shape)

    # If there are photoionisation axes add these as well
    if len(variable_photoionisation_params) > 0:
        new_axes += photoionisation_axes
        new_axes_values |= variable_photoionisation_params

    print(new_axes)
    for key, value in new_axes_values.items():
        print(key, value)

    # Create empty synthesizer grid, this returns a GridFile object we can
    # add spectra and lines to below
    new_grid, incident_grid = create_empty_grid(
        grid_dir,
        incident_grid_name,
        new_grid_file,
        new_axes_values,
        fixed_photoionisation_params,
    )

    # Check cloudy runs and potentially replace them by the nearest grid point
    # if they fail.
    failed_list = check_cloudy_runs(
        new_grid_name,
        cloudy_dir,
        incident_index_list,
        photoionisation_index_list,
    )
    print("list of failed cloudy runs:", failed_list)

    # Save list of failed runs for re-running

    # If no runs have failed, go ahead and add spectra and lines.
    if len(failed_list) == 0:
        # Add spectra
        if include_spectra:
            lam, spectra = add_spectra(
                new_grid,
                new_grid_name,
                cloudy_dir,
                incident_index_list,
                photoionisation_index_list,
                new_shape,
                spec_names=("incident", "transmitted", "nebular", "linecont"),
                norm_by_q=True,
            )
            print("Added spectra")
            print(spectra.keys())
        else:
            lam = None
            spectra = None

        # Add lines
        add_lines(
            new_grid,
            new_grid_name,
            cloudy_dir,
            incident_index_list,
            photoionisation_index_list,
            new_shape,
            lam,
            spectra,
            line_type="linelist",
        )
        print("Added lines")
