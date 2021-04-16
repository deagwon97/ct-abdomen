#! /bin/bash

mkdir -p inference_result

mkdir -p inference_models

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gWA7tiGY5yJvy7PWtEHGXFKSAa5_gQkq' -O inference_models/segment_model.pth

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rI429jEvZ_A6A3xwSMG5pf0c0z0nR41N' -O inference_models/detect_model.pth

docker run --gpus all\
            -it\
            --mount type=bind,source="/home/dacon/Dacon/bdg/ct-abdomen/inference_result",target=/workspace/ct-abdomen/inference_result\
            --mount type=bind,source="/home/dacon/Dacon/bdg/ct-abdomen/inference_models",target=/workspace/ct-abdomen/inference_models\
            --mount type=bind,source="/home/dacon/Dacon/HDD_04/ct_1800/1800 데이터(N)",target=/workspace/ct-abdomen/raw_data\
            --mount type=bind,source="/home/dacon/Dacon/HDD_04/ct_1800/1800 데이터(N)/1.원본/00000004",target=/workspace/ct-abdomen/inference_input\
            --mount type=bind,source="/home/dacon/Dacon/HDD_04/data",target=/workspace/ct-abdomen/preprocess_data\
            ct-inf
            
#--mount type=bind,source="/home/dacon/Dacon/bdg/inference_models",target=/workspace/ct-abdomen/inference_models\
            