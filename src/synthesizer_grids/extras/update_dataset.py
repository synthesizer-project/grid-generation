import h5py

grid_dir = '/Users/sw376/Dropbox/Research/data/synthesizer/grids'

source_file = "bpass-2.2.1-bin_chabrier03-0.1,300.0.hdf5"
target_file = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5"

dataset_name = "star_fraction"

# Open source file and read dataset


# with h5py.File(f'{grid_dir}/{source_file}', "r") as src:
#     src.visit(print)
#     print(src[dataset_name][...])


with h5py.File(f'{grid_dir}/{source_file}', "r") as src:

    data = src[dataset_name][...]

# Open target file and update dataset

with h5py.File(f'{grid_dir}/{target_file}', "a") as tgt:

    # Delete existing dataset if present

    if dataset_name in tgt:

        del tgt[dataset_name]

    # Create updated dataset

    tgt.create_dataset(dataset_name, data=data)

print(f"Updated '{dataset_name}' in {target_file} from {source_file}")