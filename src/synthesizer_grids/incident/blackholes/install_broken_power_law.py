"""
Create a synthesizer incident grid for a broken power-law SED.
"""

import numpy as np
import yaml
from synthesizer_grids.grid_io import GridFile
from synthesizer_grids.parser import Parser
from unyt import Angstrom, Hz, c, dimensionless, erg, s


def broken_power_law(x, edges, indices, normalisations=False, normalise=True):
    """
    Function to produce a broken power-law spectrum.

    Arguments
        x (array, float)
            x-coordinate
        edges (list, float)
            List of floats describing the edges of the individual power-law
            components.
        indices (list, float)
            List of the power-law indices. Length should be one less than
            edges.
        normalisations (list, float)
            List of normalisations. Length should be one less than
            edges.
        normalise (bool)
            Flag to decide whether to normalise to unity.

    """

    y = np.zeros(len(x))

    # if normalisations not calculated, go ahead an
    if not normalisations:
        normalisations = [
            1.0,
        ]

        # calcualte remaining normalisations
        for i, (edge, ind1, ind2) in enumerate(
            zip(edges[1:], indices, indices[1:])
        ):
            norm = normalisations[i] * (edge**ind1) / (edge**ind2)
            normalisations.append(norm)

    # now construct spectra
    for e1, e2, ind, norm in zip(
        edges[0:], edges[1:], indices, normalisations
    ):
        # identify indices within the wavelength range

        s = (x >= e1) & (x < e2)

        y[s] = norm * x[s] ** ind

    # normalise
    if normalise:
        y /= np.trapz(y[::-1], x=x[::-1])

    return y


if __name__ == "__main__":
    """
    Create incident AGN spectra assuming a broken power-law model.

    $L_{\nu} = \nu^{\alpha_i}

    """

    # define indices. Where this is a single value implies this is fixed.
    # TODO: this could be set by a separate parameter file, but this model
    # is unlikely to be used much since it should be deprecated by UnifiedAGN.

    # Set up the command line arguments
    parser = Parser(description="Broken power-law AGN model creation.")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    # get the arguments
    args = parser.parse_args()

    # open the config file and extract parameters
    with open(args.config_file, "r") as file:
        parameters = yaml.safe_load(file)

    model_name = parameters["model"]
    indices = parameters["indices"]
    edges_lam = np.array(parameters["edges"]) * Angstrom

    # Model defintion dictionary
    model = {
        "name": model_name,
        "type": "agn",
        "family": "broken_power_law",
    }

    # Define the grid filename and path
    out_filename = f"{args.grid_dir}/{model_name}.hdf5"

    # Define axes names
    axes_names = [f"alpha{i+1}" for i, _ in enumerate(indices)]

    # Define axes descriptions
    axes_descriptions = {}
    for i, axis_name in enumerate(axes_names):
        axes_descriptions[axis_name] = (
            rf"The power-law slope between {edges_lam[i]} "
            rf"\le \lambda/\AA < {edges_lam[i+1]}"
        )

    # In this case the incident axes values are just the indices
    # but we also convert lists to arrays
    axes_values = [np.array(index) * dimensionless for index in indices]

    # axes dictionary, save as part of output
    axes = dict(zip(axes_names, axes_values))

    # the shape of the grid (useful for creating outputs)
    axes_shape = list([len(axis) for axis in axes_values])

    # convert the value of edges for use
    edges_nu = c / edges_lam
    edges = edges_nu.to("THz").value

    # define wavelength and frequency grid using the edges
    lam = np.arange(edges_lam.value[0], edges_lam.value[-1], 10.0) * Angstrom
    nu = c / lam
    x = nu.to("THz").value

    # create empty spectra grid
    spec = np.zeros((*axes_shape, len(lam)))

    # loop over grid points
    # TODO: this needs generalising, but not a priority
    for i, alpha1 in enumerate(indices[0]):
        for j, alpha2 in enumerate(indices[1]):
            for k, alpha3 in enumerate(indices[2]):
                # define the indices of the current model
                indices_ = [alpha1, alpha2, alpha3]

                # generate the broken power-law spectra
                spec[i] = broken_power_law(x, edges[::-1], indices_[::-1])

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename)

    # Define which axes are logged
    log_on_read = {"alpha1": False, "alpha2": False, "alpha3": False}

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model,
        axes=axes,
        descriptions=axes_descriptions,
        wavelength=lam,
        log_on_read=log_on_read,
        spectra={"incident": spec * erg / s / Hz},
        weight="bolometric_luminosities",
    )

    # Include the specific ionising photon luminosity
    print("Calculating and saving specific ionising luminosity")
    out_grid.add_specific_ionising_lum()
