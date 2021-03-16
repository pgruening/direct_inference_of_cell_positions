# Execute this script to access the Docker-container that can
# run the code.
# you need to change your data path here!
docker run -it \
    --gpus all \
    --name pr \
    --rm \
    -v $(pwd):/workingdir \
    -v /data_ssd0/gruening/killa_seg/clean_simulation_data/data:/data \
    --user $(id -u $USER):$(id -g $USER) \
    pr bash
    