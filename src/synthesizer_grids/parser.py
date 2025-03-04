"""A module for parsing command line arguments in a standardised way.

Example Usage:
    parser = Parser("A description of the program")
    parser.add_argument("--flag", "-f", action="store_true", help="A flag")
    parser.add_argument("--value", "-v", type=int, help="A value")
    args = parser.parse()

"""

import argparse


class Parser(argparse.ArgumentParser):
    """
    A class for parsing command line arguments in a standardised way.

    NOTE: This class is a wrapper around the argparse.ArgumentParser class
    adding some standardised arguments and quality of life methods.

    Attributes:
        parser (argparse.ArgumentParser):
            The parser instance.
    """

    def __init__(self, description, with_alpha=False, cloudy_args=False):
        """
        Create the parser instance.

        Args:
            description (str):
                A description of the program.
            with_alpha (bool):
                Should the parser accept alpha arguments?
        """
        # Create the parser
        super(Parser, self).__init__(description=description)

        # First add the common arguments

        # Add filepath argument for the grids.
        self.add_argument(
            "--grid-dir",
            type=str,
            help="The directory containing (or to contain) the grids.",
            required=True,
        )

        # Are we talking?
        self.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Are we talking?",
        )

        # Now add the cloudy specific arguments or incident grid arguments
        if not cloudy_args:
            self._add_incident_args()
        else:
            self._add_cloudy_args()

        # Add alpha flag if necessary
        if with_alpha:
            self._add_alpha_args()

    def _add_incident_args(self):
        """Add arguments for incident grids."""
        # Add filepath argument for the input files used to create
        # SPS incident grids.
        self.add_argument(
            "--input-dir",
            type=str,
            help="The input file path",
            required=False,
        )

        # Add download flag
        self.add_argument(
            "--download",
            "-d",
            action="store_true",
            help="Should the data be downloaded?",
        )

    def _add_cloudy_args(self):
        """Add arguments for cloudy grids."""
        # Path to directory where cloudy runs are
        self.add_argument(
            "--cloudy-output-dir",
            type=str,
            required=True,
            help="The directory containing each "
            "of the individual cloudy run outputs.",
        )

        # The name of the incident grid
        self.add_argument(
            "--incident-grid",
            type=str,
            required=True,
            help="The name of the incident grid (the grid "
            "used to generate the cloudy runs).",
        )

        # The cloudy parameters, including any grid axes
        self.add_argument(
            "--cloudy-paramfile",
            type=str,
            required=False,
            default="c23.01-sps",
            help="The path to the cloudy parameter file.",
        )

        # A second cloudy parameter set which supersedes the above
        self.add_argument(
            "--cloudy-paramfile-extra",
            type=str,
            required=False,
            help="The path to the cloudy 'extra' parameter file.",
        )

        # # Should we include the spectra in the grid?
        # self.add_argument(
        #     "--include-spectra",
        #     type=bool,
        #     action="store_true",
        #     help="Should the spectra be included in the grid?",
        # )

        # # Boolean flag as to whether to attempt to replace missing files
        # self.add_argument(
        #     "--replace",
        #     action="store_true",
        #     help="Should missing files be replaced?",
        # )

        # # Define the line calculation method.
        # self.add_argument(
        #     "--line-calc-method",
        #     type=str,
        #     default="lines",
        #     required=False,
        #     help="The method used to calculate the line fluxes "
        #     "(either 'lines' or 'linelist')",
        # )

        # # Define the machine (for rerunning cloudy runs)
        # self.add_argument(
        #     "--machine",
        #     type=str,
        #     default=None,
        #     required=False,
        #     help="The machine used to run the cloudy runs "
        #     "(currently only supports apollo)",
        # )

        # # Should we normalise by the specific ionising luminosity?
        # self.add_argument(
        #     "--norm-by-Q",
        #     "-Q",
        #     default=True,
        #     required=False,
        #     help="Should the grid be normalised by the specific "
        #     "ionising luminosity? (default is True)",
        # )

    def _add_alpha_args(self):
        """Add arguments for alpha enhancement."""
        self.add_argument(
            "--inidividual",
            "-i",
            action="store_true",
            help=(
                "Should individual grids be made for a range of "
                "alpha enhancements?"
            ),
        )
        self.add_argument(
            "--full",
            "-f",
            action="store_true",
            help="Should a 3D grid be made with an alpha enhancement axis?",
        )

    def __str__(self):
        """Summarise the command line arguments."""
        self.print_help()
        return ""
