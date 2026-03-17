## Cloudy photoionisation modelling pipeline

This set of scripts facilitates the creation of new sets of grids with photoionisation modelling. This adds additional spectra (transmission, nebular, nebular_continuum) and adds line quantities (luminosities and continuum values).

There are three steps to the process:

- First, define a workflow configuration (incident grid or Sobol sampling) and call `run_workflow` to emit the metadata, linelists, and (for Sobol) the HDF5 parameter grid that downstream steps consume.
- Next, use `run_cloudy.py` directly or via the submission helpers. The script now generates all Cloudy inputs on-the-fly for the requested incident/photoionisation indices before executing Cloudy.
- Finally, `create_synthesizer_grid.py` gathers the `cloudy` outputs and produces a new grid.

### Preparing Cloudy metadata

`workflow.py` consolidates all pre-run preparation. The orchestrator reads an incident grid (or builds Sobol samples from it), resolves the Cloudy parameter files, copies the requested linelist into each incident directory, and writes a `grid_parameters.yaml` metadata file together with any required Sobol HDF5 datasets. No Cloudy `.in` files are generated at this stage; those are produced at runtime by `run_cloudy.py`.

Two entry modes are supported:

* **Incident grid sampling** – enumerate every point in an existing incident grid and combine it with a set of fixed/variable Cloudy parameters.
* **Sobol sampling** – draw continuous samples from the ranges defined in a YAML file and interpolate the incident spectra on-the-fly.

Example usage (see also `examples/run_incident_workflow_example.py` for a
ready-to-run script):

```python
from synthesizer_grids.cloudy.config import (
  CloudyParameterSet,
  CloudyWorkflowConfig,
  IncidentSamplerConfig,
)
from synthesizer_grids.cloudy.workflow import run_workflow

params = CloudyParameterSet.from_files("params/c23.01-sps-grid.yaml")
config = CloudyWorkflowConfig(
  mode="incident",
  incident=IncidentSamplerConfig(
    incident_grid="bc03_salpeter",
    incident_grid_dir="/path/to/grids",
    cloudy_output_dir="/path/to/cloudy/outputs",
    parameter_files=params,
  ),
)
run_workflow(config)
```

Switching to Sobol sampling requires instantiating `SobolSamplerConfig` instead and pointing at the desired parameter YAML plus sample count/seed. Ready-made Cloudy parameter files live in the `params/` directory and can be combined using `CloudyParameterSet.from_files` to override defaults. See `examples/run_sobol_workflow_example.py` for a CLI template.

The workflow optionally triggers one of the submission helpers (see `submission_scripts.py`) to emit a SLURM script that already references `run_cloudy.py`.

### Run `cloudy`

Next, we can use `run_cloudy.py` to automatically run either a single model, all models, or all models for a given photoionisation grid point (the suggested behaviour for coupling with an HPC array job). The script now regenerates the Cloudy `.in/.yaml` files on-the-fly for every requested index using the metadata emitted by `workflow.py`, runs Cloudy, moves the produced spectra into `output/`, and deletes transient files on success.

This behaviour depends on the choice of `--incident-index` and `--photoionisation-index`. Setting both will run a single model, setting only `--incident-index` will run all models at a particular incident grid point, while setting neither will result in all models being run in serial (not recommended except for tiny grids). Alternatively, pass `--list-file` and `--list-index` to execute arbitrary subsets.

```
python run_cloudy.py \
    --grid-name=grid_name \
    --cloudy-output-dir=/path/to/cloudy/outputs \
    --cloudy-executable-path=/path/to/cloudy/executable \
    --incident-index=0
    --photoionisation-index=0
```

### Create synthesizer grid

Finally, we need to create a new `synthesizer` grid object containing the cloudy computed lines and spectra. This is acheived by running `create_synthesizer_grid.py`.

```
python create_synthesizer_grid.py \
  --incident-grid=incident_grid_name \
  --grid-dir=/path/to/grids \
  --cloudy-output-dir=/path/to/cloudy/outputs \
  --cloudy-paramfile=c23.01-sps \
  --cloudy-paramfile-extra=test_suite/reference_ionisation_parameter
```

