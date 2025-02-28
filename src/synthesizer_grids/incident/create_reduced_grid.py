"""
Script to make a reduced grid, for example limiting the number of age bins
to a specific set or a max age.

Example:
    python create_reduced_grid.py --grid-dir path/to/grids \
        --original-grid bpass-2.2.1-bin_chabrier03-0.1,300.0 \
        --new-ages=6.,7.,8.
    python create_reduced_grid.py -grid_dir path/to/grids \
        -original_grid bpass-2.2.1-bin_chabrier03-0.1,300.0 \
        -ages=6.,7.,8, -metallicities=0.001,0.0
"""

import argparse

import h5py
import numpy as np
from synthesizer.grid import Grid

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

    parser.add_argument(
        "--new-ages",
        type=str,
        required=False,
        default=None,
        help="set of ages in years, e.g. --new-ages=6.,7.,8.",
    )

    parser.add_argument(
        "--metallicities",
        type=str,
        required=False,
        default=None,
        help="specific set of metallicities, e.g. --metallicities=0.008,0.01",
    )

    args = parser.parse_args()

    # open the parent incident grid
    original_grid = Grid(
        args.original_grid,
        grid_dir=f"{args.grid_dir}",
        read_lines=False,
    )

    print(args.new_ages)

    print(original_grid)

    if args.new_ages:
        new_ages = np.array(list(map(float, args.new_ages.split(","))))
        print(new_ages)

    if args.metallicities:
        metallicities = np.array(
            list(map(float, args.metallicities.split(",")))
        )
        print(metallicities)

    # get name of new grid
    new_grid_name = args.original_grid

    if args.new_ages:
        new_grid_name += f"-new_ages:{args.new_ages}"

        # currently args.new_ages = 100000000.0
        # but I want this to become a string with #1e8 instead
    elif args.max_age:
        new_grid_name += f"-max_age:{'{:.0e}'.format(args.max_age)}"

    if args.metallicities:
        new_grid_name += f"-metallicities:{args.metallicities}"

    # create an indices array
    all_age_indices = np.indices(original_grid.ages.shape)[0]
    all_metallicity_indices = np.indices(original_grid.metallicities.shape)[0]

    # maximum index for the age
    if args.new_ages:
        indices = []
        for age in new_ages:
            print(age)
            indices.append(np.where(original_grid.log10ages == age)[0][0])
        age_indices = np.array(indices)
    elif args.max_age:
        age_indices = all_age_indices[original_grid.log10ages <= args.max_age]
    else:
        age_indices = all_age_indices

    print(age_indices)

    if args.metallicities:
        metallicity_indices = []
        for metallicity in metallicities:
            metallicity_indices.append(
                np.where(original_grid.metallicities == metallicity)[0][0]
            )
        metallicity_indices = np.array(metallicity_indices)
    else:
        metallicity_indices = all_metallicity_indices

    print(metallicity_indices)

    # combined_indices = np.array([age_indices, metallicity_indices])
    # print(combined_indices.shape)
    # print(combined_indices)

    # open the new grid
    with h5py.File(f"{args.grid_dir}/{new_grid_name}.hdf5", "w") as hf:
        # open the original incident model grid
        with h5py.File(
            f"{args.grid_dir}/{args.original_grid}.hdf5", "r"
        ) as hf_original:
            print("-" * 50)
            print(f"ORIGINAL GRID - {args.original_grid}")
            print("-" * 50)
            print("---- attributes")
            for k, v in hf_original.attrs.items():
                print(k, v)
            print("---- groups and datasets")
            hf_original.visititems(print)

            # copy top-level attributes
            for k, v in hf_original.attrs.items():
                hf.attrs[k] = v

            # copy over Model group
            hf_original.copy("/Model", hf, "/Model")

            # Copy attributes manually
            for attr, value in hf_original["Model"].attrs.items():
                hf["Model"].attrs[attr] = value

            for k, v in hf['Model'].attrs.items():
                print(k, v)

            # copy axes, modifying the age axis
            hf["axes/metallicities"] = hf_original["axes/metallicities"][
                metallicity_indices
            ]
            hf["axes/ages"] = hf_original["axes/ages"][age_indices]

            # copy log10_specific_ionising_lum
            for ion in hf_original[
                "log10_specific_ionising_luminosity"
            ].keys():
                a = hf_original["log10_specific_ionising_luminosity"][ion][()]
                a = a[np.ix_(age_indices, metallicity_indices)]
                hf[f"log10_specific_ionising_luminosity/{ion}"] = a

            # copy wavelength grid
            hf["spectra/wavelength"] = hf_original["spectra/wavelength"][:]

            # copy spectra
            a = hf_original["spectra/incident"][()]
            a = a[np.ix_(age_indices, metallicity_indices)]
            hf["spectra/incident"] = a

        # print attributes and datasets of new grid
        print("-" * 50)
        print(f"NEW GRID - {new_grid_name}")
        print("-" * 50)
        print("---- attributes")
        for k, v in hf.attrs.items():
            print(k, v)
        print("---- groups and datasets")
        hf.visititems(print)
