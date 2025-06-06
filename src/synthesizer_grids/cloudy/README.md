## Cloudy photoionisation modelling pipeline

This set of scripts facilitates the creation of new sets of grids with photoionisation modelling. This adds additional spectra (transmission, nebular, nebular_continuum) and adds line quantities (luminosities and continuum values).

There are three steps to the process:

- First, we need to create the input files for `cloudy`. There are two approaches here: using an incident grid, or generating spectra directly from `cloudy`.
- Next, we use `run_cloudy.py` to run `cloudy`.
- Finally, `create_synthesizer_grid.py` gathers the `cloudy` outputs and produces a new grid.

The details of each of these steps are described below.

### Creating the cloudy input grid

The first step in the process is the creation of a grid of `cloudy` input files, including the incident spectra, the configuration file, and the list of lines to save. There are two potential approaches to creating cloudy input grids, using an **incident grid** (for example from an SPS model as above) or generating the grid using a cloudy shape command (e.g. blackbody). These scearnios are handled by two separate modules. 

#### Using an incident grid

To use an incident grid we can run `create_cloudy_input_grid.py` providing the incident grid name, grid directory, output directory, `cloudy` parameter file(s), a machine, and the path to the `cloudy` executable as follows:

```
create_cloudy_input_grid.py \
    --incident-grid=test \
    --grid-dir=/path/to/grids \
    --cloudy-output-dir=/path/to/cloudy/outputs \
    --cloudy-paramfile=c23.01-sps \
    --cloudy-paramfile-extra=test_suite/reference_ionisation_parameter \
    --machine=machine \
    --cloudy-executable-path=/path/to/cloudy/executable
```

##### Parameter files

An integral part of this process are the provision of a parameter file(s) which contains the photoionisation parameters. These can either be single values or lists (arrays). When a quantity is an array this adds an additional axis to the reprocessed grid. A range of ready-made parameter files are available for a range of scnearios in the `params/` directory.

It is possible to provide two parameter files, for example a default set of parameters and then a file containing a limited set of parameters to be changed. This approach is used to build a *photoionisation test suite*; this is a series of parameter files used to generate grids where we systematically vary a photoionisation parameter (e.g. the hydrogen density or ionisation parameter) or flag. 

##### Machines

If `--machine` is specified, and it is one that is recognised (e.g. Sussex's artemis system), then a job submission script will be produced. The will be an array job with each job being a single incident grid point, i.e. each jobs runs all the photoionisation models. In most cases this will be a handful of models, but for comprehensive grids this could be hundreds or even thousands of cloudy models that will take sometime to run.

#### Using a cloudy shape command

As an alternative we can create a grid directly from using one of `cloudy`'s in-built shape commands (e.g. blackbody). To do this we need to provide a yaml file containing the name of the model (at the moment this is limited to `blackbody` and `agn`) with all the parameters, including any that are to be varied as a list. **NOTE**: at present this is unlikely to be working correctly.

### Run `cloudy`

Next, we can use the `run_cloudy.py` to automatically run either a single model, all models, or all models for a given photoionisation grid point (the suggested behaviour for coupling with a HPC array job). This behaviour depends on the choice of --incident-index and --photoionisation-index.
Setting both will run a single model, setting only `--incident-index` will run all models at a particularly incident grid point, while setting neither will result in all models being run in serial (not recommended except for tiny grids).

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

