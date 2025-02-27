"""
Download BC03 and convert to HDF5 synthesizer grid.
"""

import gzip
import os
import re
import shutil
import sys
import tarfile

import numpy as np
import requests
from tqdm import tqdm
from unyt import Hz, angstrom, dimensionless, erg, s, yr
from utils import get_model_filename

from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser


def decompress_gz_recursively(directory):
    """Decompress all .gz files in a directory and its subdirectories."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".gz"):
                gz_file_path = os.path.join(root, file)
                with gzip.open(gz_file_path, "rb") as gz_file:
                    with open(gz_file_path[:-3], "wb") as decompressed_file:
                        shutil.copyfileobj(gz_file, decompressed_file)
                os.remove(gz_file_path)


def extract_and_decompress_tgz(file_path, extract_path):
    """Extract and decompress a .tgz file."""
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

    decompress_gz_recursively(extract_path)


def download_data(input_dir):
    """Download BC03 data and extract it."""
    # Define base path
    save_path = "bc03.models.padova_2000_chabrier_imf.tar.gz"

    url = (
        "http://www.bruzual.org/bc03/Original_version_2003/"
        "bc03.models.padova_2000_chabrier_imf.tar.gz"
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

    # Prompt user for files if not provided--------------------
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

    # Initialise ---------------------------------------------------------
    ageBins = None
    lambdaBins = None
    metalBins = [None] * len(files)
    seds = np.array([[[None]]])

    print("Reading BC03 files and converting...")
    # Loop SED tables for different metallicities
    for iFile, fileName in enumerate(files):
        print("Converting file ", fileName)
        with open(fileName, "r") as file:
            ages, lastLine = readBC03Array(file)  # Read age bins
            nAge = len(ages)
            print("Number of ages: %s" % nAge)
            if ageBins is None:
                ageBins = ages
                seds.resize(
                    (seds.shape[0], len(ageBins), seds.shape[2]),
                    refcheck=False,
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
                    (seds.shape[0], seds.shape[1], len(lambdaBins)),
                    refcheck=False,
                )
            if not np.array_equal(
                lambdas, lambdaBins
            ):  # check for consistency
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


def make_grid(input_dir, grid_dir, synthesizer_model_name):
    """Main function to convert BC03 grids and
    produce grids used by synthesizer.

    Arguments:
        input_dir (str)
            The directory where input files are downloaded.
        grid_dir (str)
            The directory where the grid is created.
        synthesizer_model_name (str)
            The name of the model used as the filename.
    """

    # output filename
    out_filename = f"{grid_dir}/{synthesizer_model_name}.hdf5"

    # Define base path
    basepath = f"{input_dir}/bc03/" "models/Padova2000/chabrier/"

    # Define files
    files = [
        "bc2003_hr_m122_chab_ssp.ised_ASCII",
        "bc2003_hr_m132_chab_ssp.ised_ASCII",
        "bc2003_hr_m142_chab_ssp.ised_ASCII",
        "bc2003_hr_m152_chab_ssp.ised_ASCII",
        "bc2003_hr_m162_chab_ssp.ised_ASCII",
        "bc2003_hr_m172_chab_ssp.ised_ASCII",
    ]

    out = convertBC03([basepath + s for s in files])

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
    print("Calculating and saving specific ionising luminosity")
    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="BC03 download and grid creation")

    # Unpack the arguments
    args = parser.parse_args()

    # the directory to store downloaded input files
    input_dir = args.input_dir

    # the directory to store the grid
    grid_dir = args.grid_dir

    sps_name = "bc03"

    # append sps_name to input_dir to define where to store downloaded input
    # files
    input_dir += f"/{sps_name}"

    # create directory to store downloaded output if it doesn't exist
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    # Download data if asked
    if args.download:
        download_data(input_dir)

    model = {
        "sps_name": sps_name,
        "sps_version": False,
        "sps_variant": False,
        "imf_type": "chabrier03",  # named IMF or bpl (broken power law)
        "imf_masses": [0.1, 100],
        "imf_slopes": False,
        "alpha": False,
    }

    # Create synthesizer style model name
    synthesizer_model_name = get_model_filename(model)
    print(synthesizer_model_name)

    # Get and write the grid
    out_filename = make_grid(input_dir, grid_dir, synthesizer_model_name)
