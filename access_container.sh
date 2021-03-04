#! /bin/bash

docker run --gpus all -it --mount type=bind,source="/home/dacon/Dacon/HDD_04/ct_1800/1800 데이터(N)",target=/workspace/ct-abdomen/raw_data ct-inf