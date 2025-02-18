"""A module containing I/O helper functions.

This module contains a helper class for reading and writing Synthesizer grids
to HDF5 files. This is a helper class should be used in grid generation scripts
to help with the writing of the Synthesizer format.

When intialised, a new file will always be created. This will overwrite any
existing file with the same name. However, for the use case in these scripts
that is the desirable behaviour. If this gets in your way raise an issue on the
GitHub repository.

Any subsequent interactions with the file will be done in append mode.

Common parts of the grid can be written out using the write_grid_common method.

Any non-standard datasets or attributes can be written using the write_dataset
and write_attribute methods. These can be used to write out model specific
datasets and attributes.

Example usage:

    # Create a grid file
    grid = GridFile("grid.hdf5")

    # Write out the common parts of the grid
    grid.write_grid_common(
        model={"model": "BPASS v2.2.1"},
        wavelength=wavelength,
        axes=axes,
        spectra=spectra,
    )

"""

from datetime import date

import h5py
import numpy as np
from synthesizer._version import __version__ as synthesizer_version
from synthesizer.photoionisation import Ions
from synthesizer.sed import Sed
from synthesizer.units import has_units
from tqdm import tqdm
from unyt import dimensionless, erg, s, unyt_array

from synthesizer_grids._version import __version__ as grids_version


class GridFile:
    """
    A helper obejct for reading/writing Synthesizer grids from/to HDF5 files.

    A new file will always be created when a GridFile object is created. This
    will overwrite any existing file with the same name. However, for the use
    case in these scripts that is the desirable behaviour. If this gets in
    your way raise an issue on the GitHub repository.

    Attributes:
        filepath (string)
            The file path to where the file should be stored.
        hdf (h5py._hl.files.File)
            The HDF5 file helper object from h5py that interfaces with the
            file on disk.
        mode (string)
            The mode the file is open in. This can be "r" for read, "w" for
            write, "r+" for read and write, or "a" for append.
    """

    # Define common descriptions
    descriptions = {
        "log10age": "Logged stellar population ages (dex(yr))",
        "log10ages": "Logged stellar population ages (dex(yr))",
        "age": "Stellar population ages (yr)",
        "ages": "Stellar population ages (yr)",
        "log10metallicities": "Logged stellar population metallicity",
        "log10metallicity": "Logged stellar population metallicity",
        "metallicities": "Stellar population metallicity",
        "metallicity": "Stellar population metallicity",
        "log_on_read": "Boolean, True for interpolated axes",
    }

    def __init__(self, filepath):
        """
        Initialise the helper.

        This will open the HDF5 file.

        Args:
            filepath (str)
                The file path to where the file should be stored. This should
                include the file name itself.
        """
        # Store the filepath for posterity
        self.filepath = filepath

        # Setup the HDF5 file attribute
        self.hdf = None

        # Set the mode to write by default
        self.mode = "w"

        # Create the file if it doesn't exist
        self._create_file()

    def _create_file(self):
        """
        Create the file if it doesn't exist.

        NOTE: This will overwrite the mode with and append mode.
        """
        self.hdf = h5py.File(self.filepath, self.mode)
        self.hdf.attrs["synthesizer_grids_version"] = grids_version
        self.hdf.attrs["synthesizer_version"] = synthesizer_version
        self.hdf.attrs["date_created"] = str(date.today())
        self.hdf.close()
        self.hdf = None
        self.mode = "r+"

    def _open_file(self):
        """Open the file if it isn't already open."""
        if self.hdf is None:
            self.hdf = h5py.File(self.filepath, self.mode)

    def _close_file(self):
        """Close the file if it is open."""
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None

    def _dataset_exists(self, key):
        """
        Check to see if a dataset exists already.

        Only appicable when self.mode = "r+" (i.e. appending)

        Args:
            key (str)
                The key to find. Must be the full path,
                i.e. "Group/SubGroup/Dataset".
        """
        if key not in self.hdf:
            return False
        return True

    def _attr_exists(self, group, attr_key):
        """
        Check to see if a dataset exists already.

        Only appicable when self.mode = "r+" (i.e. appending)

        Args:
            group (str)
                The key of the group where key is stored. Must be the full
                path, i.e. "Group/SubGroup".
            attr_key (str)
                The attribute key.
        """
        if attr_key not in self.hdf[group].attrs:
            return False
        return True

    def write_attribute(self, group, attr_key, data):
        """
        Write data into an attribute.

        Args:
            group (str)
                The key of the group where attr_key will be stored. Must be the
                full path, i.e. "Group/SubGroup".
            attr_key (str)
                The attribute name.
            data (array-like/str/float/int)
                The data to write. Shape and dtype will be inferred from this
                input data.
        """
        # Open the file
        self._open_file()

        # If the dataset exists already we need to throw an error (safe if
        # we are appending and overwriting as this was handled above)
        if self._attr_exists(group, attr_key):
            raise ValueError(f"{attr_key} already exists in {group}")

        # Finally, Write it!
        self.hdf[group].attrs[attr_key] = data

        self._close_file()

    def write_dataset(
        self,
        key,
        data,
        description,
        log_on_read,
        verbose=True,
        **extra_attrs,
    ):
        """
        Write data into a dataset.

        Args:
            key (str)
                The key to write data at. Must be the full path,
                i.e. "Group/SubGroup/Dataset".
            data (array-like)
                The data to write. Shape and dtype will be inferred from this
                input data.
            description (str)
                A short description of the dataset to be stored alongside
                the data.
            units (str)
                The units of this dataset. Defaults to "dimensionless". These
                should be in the same format unyt would produce.
            log_on_read (bool)
                A dictionary with Boolean values for each axis, where True
                indicates that the attribute should be interpolated in
                logarithmic space.
            verbose (bool)
                Are we talking?
            extra_attrs (dict)
                Any attributes of the dataset can be passed in the form:
                attr_key=attr_value.
        """
        # Open the file
        self._open_file()

        # If the dataset exists already we need to throw an error (safe if
        # we are appending and overwriting as this was handled above)
        if self._dataset_exists(key):
            raise ValueError(f"{key} already exists")

        # Ensure we have units on the data
        if not has_units(data):
            raise ValueError(
                f"Data for {key} has no units. Please provide units."
            )

        # Finally, Write it!
        dset = self.hdf.create_dataset(
            key,
            data=data.value,
            shape=data.shape,
            dtype=data.dtype,
        )

        # Set the units attribute
        dset.attrs["Units"] = str(data.units)

        # Include a brief description
        dset.attrs["Description"] = description

        # Write out whether we should log this dataset before using it to
        # interpolate spectra
        dset.attrs["log_on_read"] = log_on_read

        # Handle any other attributes passed as kwargs
        for dset_attr_key, val in extra_attrs.items():
            dset.attrs[dset_attr_key] = val

        self._close_file()

    def read_attribute(self, attr_key, group="/"):
        """
        Read an attribute and return it.

        Args:
            attr_key (str)
                The key to read.
            group (str)
                The key of the group where attr_key is stored. Must be the
                full path, i.e. "Group/SubGroup".

        Returns
            array-like/float/int/str
                The attribute stored at hdf[group].attrs[attr_key].
        """
        self._open_file()
        attr = self.hdf[group].attrs[attr_key]
        self._close_file()
        return attr

    def read_dataset(self, key, print_description=False, indices=None):
        """
        Read a dataset and return it.

        Args:
            key (str)
                The key to read. Must be the full path,
                i.e. "Group/SubGroup/Dataset".
            print_description (bool)
                Should we report what we read?

        Returns
            unyt_array/array-like
                The dataset stored at hdf[key].
        """
        # Open the file if necessary
        self._open_file()

        # Get the data, handling whether we get everything or a subset
        if indices is None:
            data = self.hdf[key][...]
        else:
            data = self.hdf[key][indices]

        # Get the units
        unit_str = self.hdf[key].attrs["Units"]

        # Print the description if asked
        if print_description:
            print(self.hdf[key].attrs["Description"])

        self._close_file()

        return unyt_array(data, unit_str)

    def copy_dataset(self, alt_key, key):
        """
        Make a link between an existing dataset and an alternative key.

        This make a duplicate so descriptions and units can be associated. It
        should therefore only be used for small datasets.

        Args:
            alt_key (str)
                The alternative key to copy to.
            key (str)
                The existing key to copy.
        """
        # Open the file
        self._open_file()

        # Get the data to copy
        data = self.hdf[key][...]
        des = self.hdf[key].attrs["Description"]
        units = self.hdf[key].attrs["Units"]
        log_on_read = self.hdf[key].attrs["log_on_read"]

        # Write the alternative version
        self.write_dataset(
            alt_key,
            unyt_array(data, units),
            des,
            log_on_read=log_on_read,
        )

        self._close_file()

    def write_grid_common(
        self,
        axes,
        wavelength,
        spectra,
        log_on_read,
        alt_axes=(),
        descriptions={},
        model={},
        weight="initial_masses",
    ):
        """
        Write out the common parts of a Synthesizer grid.

        This writer method writes all datasets and attributes that Synthesizer
        expects to exist, regardless of model.

        Any model specific datasets and attributes can still be written using
        explicit calls to write_dataset and write_attribute.

        Args:
            wavelength (unyt_array)
                The wavelength array of the spectra grid.
            axes (dict, unyt_array)
                A dictionary containing the arrays describing each axis of the
                grid.
            spectra (dict, unyt_array)
                A dictionary containing the spectra grids. Each key value pair
                should be {"spectra_type": spectra_grid}. "spectra_type" will
                be the key used for the dataset.
            log_on_read (dict)
                A dictionary with Boolean values for each axis, where True
                indicates that the attribute should be interpolated in
                logarithmic space.
            alt_axes (list, string)
                Alternative names for the axes. These will create soft links
                within the file. THIS WILL LIKELY BE DEPRECATED IN THE FUTURE.
            descriptions (dict, string)
                A dictionary containing a brief description of each dataset.
                Keys must match axes. Common descriptions are
                already included in the descriptions class attribute but can
                be overidden here.
            model (dict)
                A dictionary containing the metadata of the model used.
            weight (str)
                The variable to used to normalise the spectra in the grid. For
                instance, in most SPS models this will initial mass normalised,
                the Synthesizer property for this is "initial_masses". By
                default this is set to "initial_masses".

        Raises:
            ValueError
                If arguments disagree with each other an error is thrown.
        """
        if len(model) > 0:
            self.write_model_metadata(model)

        # Write out the axis names to an attribute
        self.write_attribute("/", "axes", list(axes.keys()))

        # Parse descriptions and use defaults if not given
        for key in axes:
            if key in descriptions:
                continue
            if key not in descriptions and key in self.descriptions:
                descriptions[key] = self.descriptions[key]
            else:
                raise ValueError(
                    "No description was provided for non-standard "
                    f"axis ({key}). Pass a description to the descriptions "
                    "kwarg."
                )

        # Handle any alternative axis names that have been given
        if len(alt_axes) > 0 and len(alt_axes) == len(axes):
            self.write_attribute("/", "axes_alternative", alt_axes)
        elif len(alt_axes) > 0:
            raise ValueError(
                "alt_axes were passed but conflicted with axes dict."
                f"axes.keys()={axes.keys()}, alt_axes={alt_axes}"
            )

        # Write out each axis array
        for axis_key, axis_arr in axes.items():
            # Ensure we have units on this axis
            if not has_units(axis_arr):
                raise ValueError(
                    f"Axis {axis_key} has no units. Please provide units."
                )

            # Write the dataset
            self.write_dataset(
                "axes/" + axis_key,
                axis_arr,
                descriptions[axis_key],
                log_on_read=log_on_read[axis_key],
            )

        # Write out the spectra grids
        self.write_spectra(spectra, wavelength, weight=weight)

    def write_spectra(self, spectra, wavelength, weight="initial_masses"):
        """
        Write out the spectra grids.

        This will write out the spectra grids to the file.

        Args:
            spectra (dict)
                A dictionary containing the spectra grids. Each key value pair
                should be {"spectra_type": spectra_grid}. "spectra_type" will
                be the key used for the dataset.
            wavelength (unyt_array)
                The wavelength array of the spectra grid.
            weight (str)
                The variable to used to normalise the spectra in the grid. For
                instance, in most SPS models this will initial mass normalised,
                the Synthesizer property for this is "initial_masses". By
                default this is set to "initial_masses".
        """
        # Write out the wavelength array
        self.write_dataset(
            "spectra/wavelength",
            wavelength,
            "Wavelength of the spectra grid",
            log_on_read=False,
        )

        # Store the weight variable as an attribute
        self.write_attribute("spectra", "WeightVariable", weight)

        # Write out each spectra
        for key, val in spectra.items():
            # Make sure the spectra has units
            if not has_units(val):
                raise ValueError(
                    f"Spectra {key} has no units. Please provide units."
                )

            # Write the spectra
            self.write_dataset(
                "spectra/" + key,
                val,
                "Three-dimensional spectra grid,"
                " [age, metallicity, wavelength]",
                log_on_read=False,
            )

        # calcualte bolometric luminosity and add
        self.add_bolometric_luminosity()

    def add_specific_ionising_lum(self, ions=("HI", "HeII"), limit=100):
        """
        Calculate the specific ionising photon luminosity for different ions.

        This will also write them to the file.

        This can only be used after the spectra arrays have been written out!

        Args:
            ions (list)
                A list of ions to calculate Q for.
            limit (float/int)
                An upper bound on the number of subintervals
                used in the integration adaptive algorithm.

        """
        # Open the file
        self._open_file()

        # Get the properties of the grid including the dimensions etc.
        (
            _,
            shape,
            _,
            _,
            _,
            index_list,
        ) = self.get_grid_properties()

        self._close_file()

        # Set up output arrays in a dict
        out_arrs = {}
        for ion in ions:
            out_arrs[ion] = np.zeros(shape)

        # Get wavelength grid
        lam = self.read_dataset("spectra/wavelength")

        # Loop over grid points, calculate the ionising luminosity and store it
        for indices in tqdm(index_list):
            indices = tuple(indices)

            # Loop over ions
            for ion in ions:
                # Get the ionisation energy
                ionisation_energy = Ions.energy[ion]

                # Get incident spectrum
                lnu = self.read_dataset("spectra/incident", indices=indices)

                # Calculate Q
                sed = Sed(lam, lnu)
                ionising_lum = sed.calculate_ionising_photon_production_rate(
                    ionisation_energy=ionisation_energy,
                    limit=limit,
                )

                # Store the results at the correct indices
                out_arrs[ion][indices] = np.log10(ionising_lum)

        # Loop over the iopns and write out their arrays
        for ion in ions:
            self.write_dataset(
                f"log10_specific_ionising_luminosity/{ion}",
                out_arrs[ion] * dimensionless,
                "Two-dimensional {ion} ionising photon "
                "production rate grid, [age, Z]",
                log_on_read=False,
            )

        self._close_file()

    def add_bolometric_luminosity(self):
        """
        Calculate the specific bolometric luminosity for each spectra.

        This will also write them to the file.

        This can only be used after the spectra arrays have been written out!

        Args:
            ions (list)
                A list of ions to calculate Q for.
            limit (float/int)
                An upper bound on the number of subintervals
                used in the integration adaptive algorithm.

        """
        # Open the file
        self._open_file()

        # Get the properties of the grid including the dimensions etc.
        (
            _,
            shape,
            _,
            _,
            _,
            index_list,
        ) = self.get_grid_properties()

        self._close_file()

        # output array
        bolometric_luminosities = np.zeros(shape)

        # Get wavelength grid
        lam = self.read_dataset("spectra/wavelength")

        # Loop over grid points calculate bolometric luminosity and store it
        for indices in tqdm(index_list):
            indices = tuple(indices)

            # Get incident spectrum
            lnu = self.read_dataset("spectra/incident", indices=indices)

            # Define sed
            sed = Sed(lam, lnu)

            # Store the results at the correct indices
            bolometric_luminosities[indices] = np.log10(
                sed.bolometric_luminosity.to("erg/s").value
            )

        # Loop over the iopns and write out their arrays
        self.write_dataset(
            "log10bolometric_luminosity",
            bolometric_luminosities * erg / s,
            "Bolometric luminosity grid, [age, Z]",
            log_on_read=False,
        )

        self._close_file()

    def write_model_metadata(self, model):
        """
        Write out the model metadata.

        Args:
            model (dict)
                A dictionary containing the metadata of the model used.
        """
        # Open the file
        self._open_file()

        # Create a group for the model metadata
        grp = self.hdf.create_group("Model")

        # Write out model parameters as attributes
        for key, value in model.items():
            grp.attrs[key] = value

        self._close_file()

    def write_cloudy_metadata(self, params):
        """
        Write out the Cloudy metadata.

        Args:
            params (dict)
                A dictionary containing the metadata of the Cloudy run.
        """
        # Open the file
        self._open_file()

        # Create the CloudyParams group if it doesn't exist
        if "CloudyParams" not in self.hdf:
            self.hdf.create_group("CloudyParams")
        cloudy_grp = self.hdf["CloudyParams"]

        # Add other parameters as attributes
        for k, v in params.items():
            # If v is None then convert to string None for saving in the
            # HDF5 file.
            if v is None:
                v = "None"

            # If the parameter is a dictionary (e.g. as used for abundances)
            if isinstance(v, dict):
                print(k, v)
                # Create group for this key
                nested_grp = cloudy_grp.create_group(k)

                for k2, v2 in v.items():
                    nested_grp.attrs[k2] = v2

            else:
                cloudy_grp.attrs[k] = v

        # Close the file
        self._close_file()

    def get_grid_properties(self, verbose=False):
        """Get the properties of the grid including the dimensions etc."""
        self._open_file()

        axes = self.hdf.attrs["axes"]  # list of axes

        # dictionary of axis grid points
        axes_values = {axis: self.hdf["axes"][axis][:] for axis in axes}

        self._close_file()

        # the grid axes
        if verbose:
            print(f"axes: {axes}")

        # number of axes
        n_axes = len(axes)
        if verbose:
            print(f"number of axes: {n_axes}")

        # the shape of the grid (useful for creating outputs)
        shape = list([len(axes_values[axis]) for axis in axes])
        if verbose:
            print(f"shape: {shape}")

        # determine number of models
        n_models = np.prod(shape)
        if verbose:
            print(f"number of models to run: {n_models}")

        # create the mesh of the grid
        mesh = np.array(
            np.meshgrid(*[np.array(axes_values[axis]) for axis in axes])
        )

        # create the list of the models
        model_list = mesh.T.reshape(n_models, n_axes)
        if verbose:
            print("model list:")
            print(model_list)

        # create a list of the indices

        index_mesh = np.array(np.meshgrid(*[range(n) for n in shape]))

        index_list = index_mesh.T.reshape(n_models, n_axes)
        if verbose:
            print("index list:")
            print(index_list)

        return n_axes, shape, n_models, mesh, model_list, index_list


def read_params(param_file):
    """
    Read a parameter file and return the imported parameters.

    Args:
    param_file (str) location of parameter file

    Returns:
    parameters (object)
    """
    return __import__(param_file)
