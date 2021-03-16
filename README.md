# direct_inference_of_cell_positions

## Setup
The easiest way to quickly setup the repository is with using Docker. In the terminal,
move to the *docker* folder and run:

``` bash docker_build.sh```

to setup the docker environment.

## Data
Download the data at???
You'll need to at the location of your data to the files *docker_run.sh* and *docker_run_jupyter.sh*.
In both files, change the line:

```-v path/to/your/data:/data \ ```

to wherever you've downloaded and unzipped the data.
## Training models
You can run the training within a docker container. First, run:

``` bash docker_run.sh```

To open a terminal within the container. Second, start the training with

``` python exe_train_models.py```

This will start training for the ResNet-18. If you want to train another model, change the strings in the *USED_MODELS* list. If your machine has more than one GPU, update the list *AVAILABLE_GPUS*. Several experiments will be run in parallel.

During training, a folder is created in *experiments*. It contains the *model.pt* file, as well as the training log (*seg_log_model.json*), the passed options for the model (*seg_opt.json*), and a list of which images were used for training, and which for validation (*split.json*).

## Segmentation of the test images
