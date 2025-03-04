# Synthesizer cloudy photoionisation modelling pipeline

This set of scripts facilitates the creation of new sets of grids with photoionisation modelling. This adds additional spectra (transmission, nebular, nebular_continuum) and adds line quantities (luminosities and continuum values).

There are free steps to the process:

- First, we we use `create_cloudy_input_grid.py` to create the input files for `cloudy`.
- Next, we use `run_cloudy.py` to run `cloudy`. 
- Finally, `create_synthesizer_grid.py` gathers the `cloudy` outputs and produces a new grid. 

The details of each of these steps are described below.

## Creating the cloudy input grid

The first step in the process is the creation of a grid of `cloudy` input files, including the incident spectra, the configuration file, and the list of lines to save.

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

An integral part of this process are the provision of a configuration file which contains the photoionisation parameters. These can either be single values or lists (arrays). When a quantity is an array this adds an additional axis to the reprocessed grid. A range of ready-made configuration files are available for a range of scnearios.

If `--machine` is specified, and it is one that is recognised (e.g. Sussex's artemis system), then a job submission script will be produced. The will be an array job with each job being a single incident grid point, i.e. each jobs runs all the photoionisation models. In most cases this will be a handful of models, but for comprehensive grids this could be hundreds or even thousands of cloudy models that will take sometime to run.

## Run `cloudy`

Next, we can use the `run_cloudy.py` to automatically run either a single model, all models, or all models for a given photoionisation grid point (the suggested behaviour for coupling with a HPC array job.). This behaviour depends on the choice of --incident-index and --photoionisation-index.
Setting both will run a single model, setting only `--incident-index` will run all models at a particularly incident grid point, while setting neither will result in all models being run in serial (not recommended except for tiny grids).

``` 
python run_cloudy.py \
    --grid-name=test_cloudy-c23.01-sps \
    --cloudy-output-dir=/path/to/cloudy/outputs \
    --cloudy-executable-path=/path/to/cloudy/executable \
    --incident-index=0
    --photoionisation-index=0
``` 

## Create synthesizer grid

```
python create_synthesizer_grid.py \
  --incident-grid=test \
  --grid-dir=/Users/sw376/Dropbox/Research/data/synthesizer/grids \
  --cloudy-output-dir=/path/to/cloudy/outputs \
  --cloudy-paramfile=c23.01-sps \
  --cloudy-paramfile-extra=test_suite/reference_ionisation_parameter
```

