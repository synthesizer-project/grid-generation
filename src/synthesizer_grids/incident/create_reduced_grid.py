"""
Script to make a reduced grid, for example limiting the number of age bins
to a specific set or a max age.

Example:
    python create_reduced_grid.py -grid_dir grids \
        -original_grid bpass-2.2.1-bin_chabrier03-0.1,300.0 -ages=6.,7.,8.
    python create_reduced_grid.py -grid_dir grids \
        -original_grid bpass-2.2.1-bin_chabrier03-0.1,300.0 \
        -ages=6.,7. -metallicities=0.001,0.0
"""

import argparse

import numpy as np
from synthesizer.grid import Grid
from unyt import Hz, erg, s, yr

from synthesizer_grids.grid_io import GridFile


def reduce_grid(original_grid, **axes):
    """
    Reduce the size of a grid by sampling the original axes.

    This will use a NGP sampling method to match the input axes to those
    on the original grid.
    """
    # Get all the axes on the original grid
    orig_axes_names = original_grid.axes

    # Loop over axes and get the indices for the reduction (any axis not in the
    # input axes will be kept as is)
    new_axes_mask = {
        axis: np.zeros(original_grid.shape[ind], dtype=bool)
        for ind, axis in enumerate(orig_axes_names)
    }

    for ind, axis in enumerate(orig_axes_names):
        if axis not in axes:
            new_axes_mask[axis] = np.ones(original_grid.shape[ind], dtype=bool)
        else:
            print(f"Reducing axis {axis}")
            values = axes[axis]
            for value in values:
                nearest_index = np.argmin(
                    np.abs(getattr(original_grid, axis) - value)
                )
                new_axes_mask[axis][nearest_index] = True

    # Get the new axes
    new_axes = {
        axis: getattr(original_grid, axis)[mask]
        for axis, mask in new_axes_mask.items()
    }

    # Get the spectra
    new_spectra = {
        spec_type: original_grid.spectra[spec_type] * erg / s / Hz
        for spec_type in original_grid.spectra
    }

    # Apply the masks along each axis of the spectra grid
    for i, axis in enumerate(orig_axes_names):
        for spec_type in original_grid.spectra:
            new_spectra[spec_type] = np.take(
                new_spectra[spec_type],
                np.nonzero(new_axes_mask[axis])[0],
                axis=i,
            )

    # Set up the new grid output
    new_name = original_grid.grid_name + "_reduced"
    out_path = f"{original_grid.grid_dir}/{new_name}.hdf5"
    print(f"Writing reduced grid to {out_path}")

    # Collect model metadata
    model_metadata = {k: v for k, v in original_grid._model_metadata.items()}

    # Which axes should be logged when read?
    if hasattr(original_grid, "_logged_axes"):
        log_on_read = {
            axis: log
            for axis, log in zip(orig_axes_names, original_grid._logged_axes)
        }
    else:
        log_on_read = {
            axis: "log10" in log
            for axis, log in zip(orig_axes_names, original_grid._extract_axes)
        }

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_path)

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model_metadata,
        axes=new_axes,
        wavelength=original_grid.lam,
        spectra=new_spectra,
        log_on_read=log_on_read,
        weight=original_grid.weightvariable,
    )

    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce a grid")

    parser.add_argument(
        "--grid-dir",
        type=str,
        required=True,
        help="path to grids",
    )

    parser.add_argument(
        "--original-grid",
        type=str,
        required=True,
        help="the name of the original_grid",
    )

    parser.add_argument(
        "--max-age",
        type=float,
        required=False,
        default=None,
        help="max age in years",
    )

    # Create an argument which will store new ages and metallicities in a list
    parser.add_argument(
        "--ages",
        nargs="+",
        type=float,
        required=False,
        default=None,
        help="Specific ages in yrs to use along the age axis "
        "(takes a list, e.g. --ages 1e6 1e7 1e8)",
    )
    parser.add_argument(
        "--metallicities",
        nargs="+",
        type=float,
        required=False,
        default=None,
        help="Specific metallicities to use along the metallicity axis "
        "(takes a list, e.g. --metallicities 0.001 0.01 0.02)",
    )

    args = parser.parse_args()

    # open the parent incident grid
    original_grid = Grid(
        args.original_grid,
        grid_dir=f"{args.grid_dir}",
        read_lines=False,
    )

    # Initialise the new axes
    ages = None
    metallicities = None

    if args.ages is not None:
        ages = np.array(args.ages) * yr
        print(f"New ages: {ages}")

    elif args.metallicities:
        metallicities = np.array(
            args.metallicities,
        )
        print(f"New metallicities: {metallicities}")

    elif args.max_age:
        ages = original_grid.ages[original_grid.ages <= args.max_age]

    else:
        raise ValueError("No reduction for any axis specified")

    # We can also impose a maximum age if new_age were set... bit pointless
    # but it's possible
    if args.max_age:
        ages = ages[ages <= args.max_age]

    # Collect which axes we are modifying
    axes = {}
    if ages is not None:
        axes["ages"] = ages
    if metallicities is not None:
        axes["metallicities"] = metallicities

    # Just do it...
    reduce_grid(original_grid, **axes)
