"""
Download BC03 and convert to HDF5 synthesizer grid.

There are three variants (using different atmospheres) and three different
IMFs. These are all processed automatically.

Example:
    python install_bc03-2016.py \
    --input-dir path/to/input/dir \
    --grid-dir path/to/grid/dir \

"""

import glob
import gzip
import os
import re
import shutil
import sys
import tarfile

import numpy as np
import requests
from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser
from tqdm import tqdm
from unyt import Hz, angstrom, dimensionless, erg, s, yr
from utils import get_model_filename


def extract_and_decompress_ised_files(directory):
    """Walk through the extracted directory and extract the ised files to the
    top directory and delete anything else.
    """

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ised.gz"):
                gz_file_path = os.path.join(root, file)
                new_file_path = os.path.join(input_dir, file)
                with gzip.open(gz_file_path, "rb") as gz_file:
                    with open(new_file_path[:-3], "wb") as decompressed_file:
                        shutil.copyfileobj(gz_file, decompressed_file)

    # delete all files but these
    for root, dirs, files in os.walk(directory):
        if root.split("/")[-1] not in ["bc03-2016", "bc03", "src"]:
            shutil.rmtree(root)


def extract_and_decompress_tgz(file_path, extract_path):
    """Extract and decompress the parent .tgz file. Subsequently calles
    extract_and_decompress_ised_files() to only extract the required ised
    files and delete everything else."""

    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

    extract_and_decompress_ised_files(extract_path)


def extract_tar(file_path, extract_path):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_path)


def download_data(variant, imf_type, input_dir):
    # Define base path
    save_path = f"{input_dir}/BC03_{variant.lower()}_{imf_type}.tgz"

    # Define the url
    url = (
        "http://www.bruzual.org/bc03/Updated_version_2016/"
        f"BC03_{variant.lower()}_{imf_type}.tgz"
    )

    # Call the server and get the response
    response = requests.get(url, stream=True)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    # Stream the file to disk with a nice progress bar.
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)

    # Untar the file and uncompress the contents
    extract_and_decompress_tgz(save_path, input_dir)
    os.remove(save_path)


def download_utility(input_dir):
    """
    Download the GALEV fortran bundle.
    """

    # Define base path
    save_path = f"{input_dir}/src.tar"

    # Define the url
    url = "http://www.bruzual.org/bc03/Updated_version_2016/src.tgz"

    # Call the server and get the response
    response = requests.get(url, stream=True)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    # Stream the file to disk with a nice progress bar.
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)

    # Untar the file and uncompress the contents
    extract_tar(save_path, input_dir)
    os.remove(save_path)


def compile_utility(input_dir):
    """Compile the ascii_ised (which converts the binary .ised files to ascii)
    tool."""

    source_dir = f"{input_dir}/bc03/src"
    os.system(f"make -C {source_dir} ascii_ised")


def convert_to_ascii(input_dir):
    """
    Using the ascii_ised tool convert all the .ised files in a given directory
    to ascii.
    """

    for file in glob.glob(f"{input_dir}/*.ised"):
        os.system(f"{input_dir}/bc03/src/ascii_ised {file}")


def readBC03Array(file, lastLineFloat=None):
    """Read a record from bc03 ascii file. The record starts with the
       number of elements N and is followed by N numbers. The record may
       or may not start within a line, i.e. a line need not necessarily
       start with a record.
    Parameters:
    ----------------------------------------------------------------------
    file: handle on open bc03 ascii file
    lastLineFloat: still open line from last line read, in case of a
                   record starting mid-line.
    Returns array, lastLine, where:
    ----------------------------------------------------------------------
    array = The array values read from the file
    lastLine = The remainder of the last line read (in floating format),
               for continued reading of the file
    """

    if lastLineFloat is None or len(lastLineFloat) == 0:
        # Nothing in last line, so read next line
        line = file.readline()
        lineStr = line.split()
        lastLineFloat = [float(x) for x in lineStr]
    # Read array 'header' (i.e. number of elements)
    arrayCount = int(lastLineFloat[0])  # Length of returned array
    array = np.empty(arrayCount)  # Initialise the array
    lastLineFloat = lastLineFloat[1 : len(lastLineFloat)]
    iA = 0  # Running array index
    while True:  # Read numbers until array is full
        for iL in range(0, len(lastLineFloat)):  # Loop numbers in line
            array[iA] = lastLineFloat[iL]
            iA = iA + 1
            if iA >= arrayCount:  # Array is full so return
                return array, lastLineFloat[iL + 1 :]
        line = file.readline()  # Went through the line so get the next one
        lineStr = line.split()
        lastLineFloat = [float(x) for x in lineStr]


def convertBC03(files=None):
    """Convert BC03 outputs
    Parameters (user will be prompted for those if not present):
    ----------------------------------------------------------------------
    files: list of each BC03 SED ascii file, typically named
           bc2003_xr_mxx_xxxx_ssp.ised_ASCII
    """

    # Prompt user for files if not provided
    if files is None:
        print(
            "Please write the model to read",
        )
        files = []
        while True:
            filename = input("filename >")
            if filename == "":
                break
            files.append(filename)
        print("ok checking now")
        if not len(files):
            print("No BC03 files given, nothing do to")
            return

    # Initialise
    ageBins = None
    lambdaBins = None
    metalBins = [None] * len(files)
    seds = np.array([[[None]]])

    print("Reading BC03 files and converting...")
    # Loop SED tables for different metallicities
    for iFile, fileName in enumerate(files):
        print("Converting file ", fileName)
        file = open(fileName, "r")
        # file = gzip.open(f'{fileName}.gz', 'rb')

        ages, lastLine = readBC03Array(file)  # Read age bins
        nAge = len(ages)
        print("Number of ages: %s" % nAge)
        if ageBins is None:
            ageBins = ages
            seds.resize(
                (seds.shape[0], len(ageBins), seds.shape[2]), refcheck=False
            )
        if not np.array_equal(ages, ageBins):  # check for consistency
            print("Age bins are not identical everywhere!!!")
            print("CANCELLING CONVERSION!!!")
            return
        # Read four (five ?) useless lines
        line = file.readline()
        line = file.readline()
        line = file.readline()
        line = file.readline()
        line = file.readline()

        print(line)
        # These last three lines are identical and contain the metallicity
        (jmetal,) = re.search(r"Z=([0-9]+\.?[0-9]*)", line).groups()
        metalBins[iFile] = eval(jmetal)
        seds.resize(
            (len(metalBins), seds.shape[1], seds.shape[2]), refcheck=False
        )
        # Read wavelength bins
        lambdas, lastLine = readBC03Array(file, lastLineFloat=lastLine)
        if lambdaBins is None:  # Write wavelengths to sed file
            lambdaBins = lambdas
            seds.resize(
                (seds.shape[0], seds.shape[1], len(lambdaBins)), refcheck=False
            )
        if not np.array_equal(lambdas, lambdaBins):  # check for consistency
            print("Wavelength bins are not identical everywhere!!!")
            print("CANCELLING CONVERSION!!!")
            return
        # Read luminosities
        for iAge in range(0, nAge):
            lums, lastLine = readBC03Array(file, lastLineFloat=lastLine)
            if len(lums) != len(lambdaBins):
                print("Inconsistent number of wavelength bins in BC03")
                print("STOPPING!!")
                return
            # Read useless array
            tmp, lastLine = readBC03Array(file, lastLineFloat=lastLine)
            seds[iFile, iAge] = lums
            progress = (iAge + 1) / nAge
            sys.stdout.write(
                "\rProgress: [{0:50s}] {1:.1f}%".format(
                    "#" * int(progress * 50), progress * 100
                )
            )
        print(" ")
        lastLine = None

    return (
        np.array(seds, dtype=np.float64),
        np.array(metalBins, dtype=np.float64),
        np.array(ageBins, dtype=np.float64),
        np.array(lambdaBins, dtype=np.float64),
    )


def make_grid(variant, imf_type, input_dir, out_filename):
    """Main function to convert BC03 grids and
    produce grids used by synthesizer"""

    # Define imf code
    if imf_type == "salpeter":
        imf_code = "salp"
    if imf_type == "chabrier":
        imf_code = "chab"
    if imf_type == "kroupa":
        imf_code = "kroup"

    # Define variant code
    if variant == "BaSeL":
        variant_code = "lr_BaSeL"
    if variant == "Miles":
        variant_code = "hr_xmiless"
    if variant == "Stelib":
        variant_code = "hr_stelib"

    # Define files
    files = [
        f"bc2003_{variant_code}_m22_{imf_code}_ssp.ised_ASCII",
        f"bc2003_{variant_code}_m32_{imf_code}_ssp.ised_ASCII",
        f"bc2003_{variant_code}_m42_{imf_code}_ssp.ised_ASCII",
        f"bc2003_{variant_code}_m52_{imf_code}_ssp.ised_ASCII",
        f"bc2003_{variant_code}_m62_{imf_code}_ssp.ised_ASCII",
        f"bc2003_{variant_code}_m72_{imf_code}_ssp.ised_ASCII",
        f"bc2003_{variant_code}_m82_{imf_code}_ssp.ised_ASCII",
    ]

    out = convertBC03([input_dir + "/" + s for s in files])

    metallicities = out[1]

    ages = out[2]
    ages[0] = 1e5

    lam = out[3]
    nu = 3e8 / (lam * 1e-10)

    spec = out[0]

    spec = np.swapaxes(spec, 0, 1)  # make (age, metallicity, wavelength)

    spec *= 3.826e33  # erg s^-1 AA^-1 Msol^-1
    spec *= lam / nu  # erg s^-1 Hz^-1 Msol^-1

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename)

    # A dictionary with Boolean values for each axis, where True
    # indicates that the attribute should be interpolated in
    # logarithmic space.
    log_on_read = {"ages": True, "metallicities": False}

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=lam * angstrom,
        spectra={"incident": spec * erg / s / Hz},
        alt_axes=("ages", "metallicities"),
        log_on_read=log_on_read,
        weight="initial_masses",
    )

    # Include the specific ionising photon luminosity
    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="BC03-2016 download and grid creation")

    # Unpack the arguments
    args = parser.parse_args()

    # the directory to store downloaded input files
    input_dir = args.input_dir

    # the directory to store the grid
    grid_dir = args.grid_dir

    sps_name = "bc03-2016"

    # append sps_name to input_dir to define where to store downloaded input
    # files
    input_dir += f"/{sps_name}"

    # create directory to store downloaded output if it doesn't exist
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    default_model = {
        "sps_name": sps_name,
        "sps_version": False,
        "sps_variant": False,
        "imf_type": False,  # named IMF or bpl (broken power law)
        "imf_masses": [0.1, 100],
        "imf_slopes": False,
        "alpha": False,
    }

    # download BC03 utility
    download_utility(input_dir)

    # compile BC03 utility
    compile_utility(input_dir)

    variants = ["BaSeL", "Miles", "Stelib"]

    # do not bother with the top-heavy IMF it has a different naming scheme
    # not worth the hassle.
    imf_types = ["kroupa", "salpeter", "chabrier"]

    if args.download:
        # download all the variant and IMF types desired
        for variant in variants:
            for imf_type in imf_types:
                download_data(variant, imf_type, input_dir)

        # Convert the binary ised files to ascii
        convert_to_ascii(input_dir)

    for variant in variants:
        for imf_type in imf_types:
            model_variations = {"sps_variant": variant, "imf_type": imf_type}

            model = default_model | model_variations

            synthesizer_model_name = get_model_filename(model)
            print(synthesizer_model_name)

            # this is the full path to the ultimate HDF5 grid file
            out_filename = f"{grid_dir}/{synthesizer_model_name}.hdf5"

            make_grid(variant, imf_type, input_dir, out_filename)
