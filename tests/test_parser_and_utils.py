from pathlib import Path

from synthesizer_grids.cloudy.utils import get_cloudy_params
from synthesizer_grids.parser import Parser


def test_parser_exposes_alpha_flags(tmp_path):
    parser = Parser("alpha test", with_alpha=True)

    args = parser.parse_args(
        [
            "--grid-dir",
            str(tmp_path),
            "--input-dir",
            str(tmp_path),
            "--full",
        ]
    )

    assert args.grid_dir == str(tmp_path)
    assert args.input_dir == str(tmp_path)
    assert args.full
    assert not args.inidividual


def test_parser_cloudy_arguments(tmp_path):
    parser = Parser("cloudy test", cloudy_args=True)
    cloudy_dir = tmp_path / "cloudy"

    args = parser.parse_args(
        [
            "--grid-dir",
            str(tmp_path),
            "--cloudy-output-dir",
            str(cloudy_dir),
            "--incident-grid",
            "incident-grid",
        ]
    )

    assert args.cloudy_output_dir == str(cloudy_dir)
    assert args.incident_grid == "incident-grid"
    assert args.cloudy_paramfile == "c23.01-sps"
    assert not hasattr(args, "download")


def test_get_cloudy_params_splits_fixed_and_grid(tmp_path):
    param_dir = tmp_path / "params"
    param_dir.mkdir()
    param_file = param_dir / "custom.yaml"

    param_file.write_text(
        "\n".join(
            [
                "cloudy_version: c23.01",
                "hydrogen_density: [1, 2]",
                "abundance_scalings:",
                "  carbon_to_oxygen: [0.1, 0.2]",
                "  nitrogen_to_oxygen: 0.3",
                "fixed_value: 42",
            ]
        )
    )

    fixed, grid = get_cloudy_params(
        param_file=Path(param_file).name,
        param_dir=str(param_dir),
    )

    assert fixed["cloudy_version"] == "c23.01"
    assert fixed["fixed_value"] == 42
    assert fixed["abundance_scalings.nitrogen_to_oxygen"] == 0.3

    assert grid["hydrogen_density"].tolist() == [1.0, 2.0]
    assert grid["abundance_scalings.carbon_to_oxygen"].tolist() == [0.1, 0.2]
