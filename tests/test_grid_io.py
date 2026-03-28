import h5py
import numpy as np
import pytest
from unyt import Angstrom, Hz, erg, s, unyt_array, yr

from synthesizer_grids.grid_io import GridFile


def _decode_attr(value):
    """Helper to decode HDF5 string attributes."""
    return value.decode() if isinstance(value, bytes) else value


@pytest.fixture
def simple_grid(tmp_path):
    out_path = tmp_path / "grid.hdf5"
    grid = GridFile(str(out_path))

    axes = {
        "ages": unyt_array([1.0, 10.0], yr),
        "metallicities": unyt_array([0.001, 0.02], "dimensionless"),
    }
    log_on_read = {"ages": True, "metallicities": False}
    wavelength = unyt_array([500.0, 1000.0, 2000.0], Angstrom)
    spectra_shape = (
        len(axes["ages"]),
        len(axes["metallicities"]),
        len(wavelength),
    )
    spectra = {
        "incident": unyt_array(
            np.ones(spectra_shape),
            erg / s / Hz,
        )
    }

    grid.write_grid_common(
        axes=axes,
        wavelength=wavelength,
        spectra=spectra,
        log_on_read=log_on_read,
        model={"model": "unit-test"},
    )

    return {
        "grid": grid,
        "path": out_path,
        "axes": axes,
        "wavelength": wavelength,
        "spectra_shape": spectra_shape,
    }


def test_gridfile_sets_metadata(tmp_path):
    out_path = tmp_path / "grid.hdf5"

    GridFile(str(out_path))

    with h5py.File(out_path, "r") as hdf:
        attrs = {key: _decode_attr(val) for key, val in hdf.attrs.items()}
        assert "synthesizer_grids_version" in attrs
        assert "synthesizer_version" in attrs
        assert "date_created" in attrs


def test_write_dataset_requires_units(tmp_path):
    out_path = tmp_path / "grid.hdf5"
    grid = GridFile(str(out_path))

    with pytest.raises(ValueError):
        grid.write_dataset(
            key="axes/ages",
            data=np.array([1, 2]),
            description="unitless ages should fail",
            log_on_read=False,
        )


def test_write_grid_common_writes_expected_structure(simple_grid):
    with h5py.File(simple_grid["path"], "r") as hdf:
        axes_attr = list(map(_decode_attr, hdf.attrs["axes"]))
        assert axes_attr == ["ages", "metallicities"]
        assert hdf.attrs["WeightVariable"] == "initial_masses"

        ages_dataset = hdf["axes"]["ages"]
        assert ages_dataset.shape == (2,)
        assert ages_dataset.attrs["log_on_read"]

        metallicities_dataset = hdf["axes"]["metallicities"]
        assert not metallicities_dataset.attrs["log_on_read"]

        assert hdf["spectra"]["wavelength"].shape == (3,)
        assert hdf["spectra"]["incident"].shape == simple_grid["spectra_shape"]

        model_group = hdf["Model"]
        assert model_group.attrs["model"] == "unit-test"


def test_write_lines_creates_expected_datasets(simple_grid):
    grid = simple_grid["grid"]
    axes = simple_grid["axes"]
    n_lines = 2
    base_shape = (
        len(axes["ages"]),
        len(axes["metallicities"]),
        n_lines,
    )

    lines = {
        "wavelength": unyt_array([4861.0, 6563.0], Angstrom),
        "id": ["Hbeta", "Halpha"],
        "luminosity": unyt_array(np.ones(base_shape), erg / s),
        "continuum": unyt_array(np.full(base_shape, 2.0), erg / s),
    }

    grid.write_lines(lines)

    with h5py.File(simple_grid["path"], "r") as hdf:
        assert hdf["lines"]["wavelength"].shape == (n_lines,)
        assert hdf["lines"]["luminosity"].shape == base_shape
        assert hdf["lines"]["continuum"].shape == base_shape
        assert list(map(_decode_attr, hdf["lines"]["id"][...])) == [
            "Hbeta",
            "Halpha",
        ]


def test_add_specific_ionising_luminosity(simple_grid):
    grid = simple_grid["grid"]
    axes = simple_grid["axes"]

    grid.add_specific_ionising_lum(ions=("HI",))

    with h5py.File(simple_grid["path"], "r") as hdf:
        data = hdf["log10_specific_ionising_luminosity"]["HI"][...]

    expected_shape = (
        len(axes["ages"]),
        len(axes["metallicities"]),
    )
    assert data.shape == expected_shape
    assert np.isfinite(data).all()
