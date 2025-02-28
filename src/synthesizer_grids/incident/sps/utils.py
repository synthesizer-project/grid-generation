import os
import sys

import numpy as np

# import function from incident_utils module
# allow it to see anything two folders up so that grid_utils and
# incident_utils can be accessed
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# import incident_utils

# __tag__ = incident_utils.__tag__


def get_model_filename(model):
    synthesizer_model_name = f'{model["sps_name"]}'

    if model["sps_version"] is not False:
        synthesizer_model_name += f'-{model["sps_version"]}'

    if model["sps_variant"] is not False:
        synthesizer_model_name += f'-{model["sps_variant"]}'

    # Mass limits label
    mass_limits_label = ",".join(
        map(lambda x: str(np.round(x, 2)), model["imf_masses"])
    )

    synthesizer_model_name += f'_{model["imf_type"]}-{mass_limits_label}'

    # IMF slope label
    if model["imf_type"] == "bpl":
        imf_slopes_label = ",".join(
            map(lambda x: str(np.round(x, 2)), model["imf_slopes"])
        )
        synthesizer_model_name += "-" + imf_slopes_label

    # If alpha varies then say so
    if model["alpha"] is not False:
        synthesizer_model_name += f'_alpha{model["alpha"]}'

    # If high mass slope varies then say so
    if "high_mass_slopes" in list(model.keys()):
        synthesizer_model_name += '-variable-high-mass-slope'

    return synthesizer_model_name
