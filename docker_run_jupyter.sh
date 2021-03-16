# Execute this script to run a jupyter notebook inside the Docker-container
# that can run the code.
docker run \
    --gpus all \
    --name dl \
    # change your data path here
    --rm -p 8888:8888 \
    -v $(pwd):/workingdir \
    -v /nfshome/heldt/Documents/data:/data \
    pr jupyter notebook . \
    --ip=0.0.0.0 \
    --no-browser \
    --allow-root 