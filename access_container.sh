#! /bin/bash

mkdir -p inference_result

mkdir -p inference_models

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NpJ5OUC8mhceO2Y2x9AHT3koNdBsr9TZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NpJ5OUC8mhceO2Y2x9AHT3koNdBsr9TZ" -O inference_models/segment_model.pth && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rI429jEvZ_A6A3xwSMG5pf0c0z0nR41N' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rI429jEvZ_A6A3xwSMG5pf0c0z0nR41N" -O inference_models/detect_model.pth && rm -rf /tmp/cookies.txt


docker run --gpus all\
            -it\
            --mount type=bind,source="/home/dacon/Dacon/bdg/ct-abdomen/inference_result",target=/workspace/ct-abdomen/inference_result\
            --mount type=bind,source="/home/dacon/Dacon/bdg/ct-abdomen/inference_models",target=/workspace/ct-abdomen/inference_models\
            --mount type=bind,source="/home/dacon/Dacon/HDD_04/ct_1800/1800 데이터(N)",target=/workspace/ct-abdomen/raw_data\
            --mount type=bind,source="/home/dacon/Dacon/HDD_04/ct_1800/1800 데이터(N)/1.원본/00000004",target=/workspace/ct-abdomen/inference_input\
            --mount type=bind,source="/home/dacon/Dacon/HDD_04/data",target=/workspace/ct-abdomen/preprocess_data\
            ct-inf
            
#--mount type=bind,source="/home/dacon/Dacon/bdg/inference_models",target=/workspace/ct-abdomen/inference_models\
            