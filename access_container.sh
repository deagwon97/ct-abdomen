#! /bin/bash

docker run --gpus all\
            -it\
            --mount type=bind,source="/home/dacon/Dacon/HDD_04/ct_1800/1800 데이터(N)",target=/workspace/ct-abdomen/raw_data\
            --mount type=bind,source="/home/dacon/Dacon/HDD_04/ct_1800/1800 데이터(N)/1.원본/00000004",target=/workspace/ct-abdomen/inference_input\
            --mount type=bind,source="/home/dacon/Dacon/HDD_04/data",target=/workspace/ct-abdomen/preprocess_data\
            --mount type=bind,source="/home/dacon/Dacon/bdg/inference_models",target=/workspace/ct-abdomen/inference_models\
            ct-inf
            