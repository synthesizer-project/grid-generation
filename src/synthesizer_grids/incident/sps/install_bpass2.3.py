"""
Process BPASS v2.3 and convert to HDF5 synthesizer grid.

For BPASS v2.3 there is currently only a single variant (bin) and IMF. However
v2.3 has 5 different values of the alpha enhancement. The processing at the
moment create grid files for both individual alpha enhancements and then a 3D
grid where alpha_enhancement is an axis.

Note: there is no automatic download available so models need to be downloaded
manually from:
https://warwick.ac.uk/fac/sci/physics/research/astro/research/catalogues/bpass/

The current is currently stored in 5 separate archives - one for each different
alpha-enhancement. However, this is not convenient so it makes more sense to
create a single directory and move all the files into there. That is,

<after extracting the files>
mkdir bpass_v2.3_imf135_300
mv bpass_v2.3.a*/* bpass_v2.3_imf135_300/
rm -rf bpass_v2.3.a*

Example:
    python install_bpass2.3.py \
    --input-dir path/to/input/dir \
    --grid-dir path/to/grid/dir \
"""

import numpy as np
from unyt import Hz, angstrom, dimensionless, erg, s, yr
from utils import get_model_filename

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser


def resolve_name(original_model_name, bin, alpha=False):
    """Resolve the original BPASS model name into what we need. This is
    specific to 2.3, e.g. 'bpass_v2.3_imf135_300'"""

    # bpass_imf = original_model_name.split("_")[-1]
    bpass_imf = original_model_name.split("imf")[-1]
    print("bpass imf:", bpass_imf)
    hmc = float(bpass_imf[-3:])  # high-mass cutoff

    if bpass_imf[:4] == "chab":
        imf_type = "chabrier03"
        imf_masses = [0.1, hmc]
        imf_slopes = False

    else:
        imf_type = "bpl"
        imf_masses = [0.1, 1.0, hmc]
        imf_slopes = [1.3, np.round(float(bpass_imf[:3]) / 100 + 1, 2)]

    model = {
        "original_model_name": original_model_name,
        "sps_name": "bpass",
        "sps_version": "2.3",
        "sps_variant": bin,
        "imf_type": imf_type,  # named IMF or bpl (broken power law)
        "imf_masses": imf_masses,
        "imf_slopes": imf_slopes,
        "alpha": alpha,
    }

    print(model)

    return model, bpass_imf


def parse_ionizing_file(filename):
    """
    Parse a BPASS ionizing file.
    """
    data = np.loadtxt(filename).T
    log10ages = data[0]
    ages = 10**log10ages

    return ages


def parse_spectra_file(filename):
    """
    Parse a BPASS spectra file.
    """
    data = np.loadtxt(filename).T
    wavelength = data[0]
    spectra = data[1:]

    return wavelength, spectra


def make_single_alpha_grid(
    original_model_name, input_dir, grid_dir, ae="+00", bs="bin"
):
    """make a grid for a single alpha enhancement"""

    # convert bpass alpha code (e.g. '+02' into a numerical alpha e.g. 0.2)
    alpha = float(ae) / 10.0

    # returns a dictionary containing the sps model parameters
    model, bpass_imf = resolve_name(original_model_name, bs, alpha=alpha)

    print(bpass_imf)

    # generate the synthesizer_model_name
    synthesizer_model_name = get_model_filename(model)

    print(synthesizer_model_name)

    # this is the full path to the ultimate HDF5 grid file
    out_filename = f"{grid_dir}/{synthesizer_model_name}.hdf5"

    # input directory of this specific bpass model (hence the trailing "_")
    input_dir_ = f'{input_dir}/{model["original_model_name"]}'

    # create metallicity grid and dictionary
    map_key_to_met = {
        "zem5": 0.00001,
        "zem4": 0.0001,
        "z001": 0.001,
        "z002": 0.002,
        "z003": 0.003,
        "z004": 0.004,
        "z006": 0.006,
        "z008": 0.008,
        "z010": 0.01,
        "z014": 0.014,
        "z020": 0.020,
        "z030": 0.030,
        "z040": 0.040,
    }
    map_met_to_key = {k: v for v, k in map_key_to_met.items()}
    metallicities = np.sort(np.array(list(map_met_to_key.keys())))

    # get ages
    fn_ = f"ionizing-{bs}-imf{bpass_imf}.a{ae}.zem5.dat"
    ages = parse_ionizing_file(f"{input_dir_}/{fn_}")

    # open spectra file
    fn_ = f"spectra-{bs}-imf{bpass_imf}.a{ae}.zem5.dat"
    wavelengths, spectra_ = parse_spectra_file(f"{input_dir_}/{fn_}")
    nu = 3e8 / (wavelengths * 1e-10)

    # number of metallicities and ages
    nmetal = len(metallicities)
    na = len(ages)

    # set up outputs

    # the ionising photon production rate
    log10_specific_ionising_lum = {}
    for ion in ["HI", "HeII"]:
        log10_specific_ionising_lum[ion] = np.zeros((na, nmetal))

    spectra = np.zeros((na, nmetal, len(wavelengths)))

    for imetal, metal in enumerate(metallicities):
        metallicity_key = map_met_to_key[metallicities[imetal]]

        # get ages and remaining fraction
        fn_ = f"ionizing-{bs}-imf{bpass_imf}.a{ae}.{metallicity_key}.dat"
        ages = parse_ionizing_file(f"{input_dir_}/{fn_}")

        # open spectra file
        fn_ = f"spectra-{bs}-imf{bpass_imf}.a{ae}.{metallicity_key}.dat"
        wavelengths, spectra_ = parse_spectra_file(f"{input_dir_}/{fn_}")

        for ia, _ in enumerate(ages):
            spec_ = spectra_[ia]  # Lsol AA^-1 10^6 Msol^-1

            # convert from Llam to Lnu
            spec_ /= 1e6  # Lsol AA^-1 Msol^-1
            spec_ *= 3.826e33  # erg s^-1 AA^-1 Msol^-1
            spec_ *= wavelengths / nu  # erg s^-1 Hz^-1 Msol^-1
            spectra[ia, imetal, :] = spec_

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename)

    # A dictionary with Boolean values for each axis, where True
    # indicates that the attribute should be interpolated in
    # logarithmic space.
    log_on_read = {"ages": True, "metallicities": True}

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=wavelengths * angstrom,
        spectra={"incident": spectra * erg / s / Hz},
        alt_axes=("log10ages", "metallicities"),
        log_on_read=log_on_read,
        weight="initial_masses",
    )

    # Include the specific ionising photon luminosity
    out_grid.add_specific_ionising_lum()


def make_full_grid(original_model_name, input_dir, grid_dir, bs="bin"):
    """make a full grid for different alpha-ehancements"""

    # returns a dictionary containing the sps model parameters
    model, bpass_imf = resolve_name(original_model_name, bs)

    # generate the synthesizer_model_name
    synthesizer_model_name = get_model_filename(model)

    # this is the full path to the ultimate HDF5 grid file
    out_filename = f"{grid_dir}/{synthesizer_model_name}.hdf5"

    # input directory of this specific bpass model (hence the trailing "_")
    input_dir_ = f'{input_dir}/{model["original_model_name"]}'

    # create metallicity grid and dictionary
    map_key_to_met = {
        "zem5": 0.00001,
        "zem4": 0.0001,
        "z001": 0.001,
        "z002": 0.002,
        "z003": 0.003,
        "z004": 0.004,
        "z006": 0.006,
        "z008": 0.008,
        "z010": 0.01,
        "z014": 0.014,
        "z020": 0.020,
        "z030": 0.030,
        "z040": 0.040,
    }
    map_met_to_key = {k: v for v, k in map_key_to_met.items()}
    metallicities = np.sort(np.array(list(map_met_to_key.keys())))
    log10metallicities = np.log10(metallicities)

    # create alpha-enhancement grid

    # list of available alpha enhancements
    alpha_enhancements = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])

    # look up dictionary for filename
    ae_to_aek = {-0.2: "-02", 0.0: "+00", 0.2: "+02", 0.4: "+04", 0.6: "+06"}

    # first metallicity
    metalk = map_met_to_key[metallicities[0]]

    # get ages and remaining fraction for first alpha-enhancement and
    # metallicity
    fn_ = f"""ionizing-{bs}-imf{bpass_imf}.a+00.{metalk}.dat"""
    ages = parse_ionizing_file(f"{input_dir_}/{fn_}")

    # open spectra file for first alpha-enhancement and
    # metallicity
    fn_ = f"""spectra-{bs}-imf{bpass_imf}.a+00.{metalk}.dat"""
    wavelengths, spectra_ = parse_spectra_file(f"{input_dir_}/{fn_}")
    nu = 3e8 / (wavelengths * 1e-10)

    na = len(ages)
    nmetal = len(log10metallicities)
    nae = len(alpha_enhancements)

    # set up outputs

    # the ionising photon production rate
    log10_specific_ionising_lum = {}
    log10_specific_ionising_lum["HI"] = np.zeros((na, nmetal, nae))
    log10_specific_ionising_lum["HeII"] = np.zeros((na, nmetal, nae))

    spectra = np.zeros((na, nmetal, nae, len(wavelengths)))

    for imetal, metal in enumerate(metallicities):
        for iae, alpha_enhancement in enumerate(alpha_enhancements):
            aek = ae_to_aek[alpha_enhancement]
            metalk = map_met_to_key[metal]

            # --- get remaining and remnant fraction
            fn_ = f"""ionizing-{bs}-imf{bpass_imf}.a{aek}.{metalk}.dat"""

            # get ages and remaining fraction
            ages = parse_ionizing_file(f"{input_dir_}/{fn_}")

            # open spectra file
            fn_ = f"""spectra-{bs}-imf{bpass_imf}.a{aek}.{metalk}.dat"""
            wavelengths, spectra_ = parse_spectra_file(f"{input_dir_}/{fn_}")

            for ia, age in enumerate(ages):
                spec_ = spectra_[ia]  # Lsol AA^-1 10^6 Msol^-1

                # --- convert from Llam to Lnu
                spec_ /= 1e6  # Lsol AA^-1 Msol^-1
                spec_ *= 3.826e33  # erg s^-1 AA^-1 Msol^-1
                spec_ *= wavelengths / nu  # erg s^-1 Hz^-1 Msol^-1

                spectra[ia, imetal, iae, :] = spec_  # Lsol AA^-1 10^6 Msol^-1

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename)

    log_on_read = {
        "ages": True,
        "metallicities": False,
        "alpha_enhancement": False,
    }

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
            "alpha_enhancement": alpha_enhancements * dimensionless,
        },
        descriptions={"alpha_enhancement": r"alpha ehanncement [\alpha/Fe]"},
        wavelength=wavelengths * angstrom,
        spectra={"incident": spectra * erg / s / Hz},
        alt_axes=("log10ages", "metallicities", "alpha_enhancements"),
        log_on_read=log_on_read,
        weight="initial_masses",
    )

    # Include the specific ionising photon luminosity
    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(
        description="BPASS_2.3 download and grid creation",
        with_alpha=True,
    )

    parser.add_argument(
        "-models",
        "--models",
        default="bpass_v2.3_imf135_300",
        type=lambda arg: arg.split(","),
    )

    # flags whether to build individual grids or the full grid.
    individual = True
    full = True

    # Unpack the arguments
    args = parser.parse_args()

    # the directory to store downloaded input files
    input_dir = args.input_dir

    # the directory to store the grid
    grid_dir = args.grid_dir

    # define sps name used to store the input files
    sps_name = "bpass"

    # append sps_name to input_dir to define where to store downloaded input
    # files
    input_dir += f"/{sps_name}"

    models = args.models

    for model in models:
        for bs in ["bin"]:  # no single star models , 'sin'
            # make a grid with a single alpha enahancement value
            if individual:
                for ae in ["-02", "+00", "+02", "+04", "+06"]:
                    print(ae)
                    out_filename = make_single_alpha_grid(
                        model, input_dir, grid_dir, ae=ae, bs=bs
                    )

            # make a full 3D grid
            if full:
                out_filename = make_full_grid(
                    model, input_dir, grid_dir, bs=bs
                )
