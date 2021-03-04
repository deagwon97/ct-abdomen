#! /bin/bash

docker run --gpus all -it --mount type=bind,source=/home/dacon/Dacon/bdg/ct-final/raw_data,target=/workdir/raw_data ct-inf