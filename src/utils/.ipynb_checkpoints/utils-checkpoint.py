import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


def create_meta_df(PATH):
    error_list = []
    images_paths = []
    error_dic = {}
    error_dic['1.원본'] = []
    error_dic['2.근육'] = []
    error_dic['3.내장지방'] = []
    error_dic['4.피하지방'] = []

    for person_idx in tqdm(os.listdir(PATH + f"1.원본/")):

        breaker = False
        for part in ['1.원본','2.근육','3.내장지방','4.피하지방']:
            try:
                if part == '1.원본':
                    fileslist = os.listdir(PATH + f"{part}/{person_idx}")
                else:
                    fileslist = os.listdir(PATH + f"{part}/tif/{person_idx}")

                if len(fileslist) < 20:
                    error_dic[part].append(person_idx)
                    breaker = True
                elif len(fileslist) > 500:
                    error_dic[part].append(person_idx)
                    breaker = True
                else:
                    pass
            except:
                error_dic[part].append(person_idx)
                breaker = True

        if breaker == True:
            if part == '1.원본':
                break
            breaker = False
            continue


        original_list = os.listdir(PATH + f"1.원본/{person_idx}")
        original_list = name_filter(original_list, 'converted')
        try:
            original_path = pd.DataFrame(
                                        {"1.원본" : original_list},
                                        index = list(map(export_num, original_list))
                                       ).sort_index()
            original_max = original_path.index.max()

            # 근육을 기준으로 생성
            muscle_list = os.listdir(PATH + f"2.근육/tif/{person_idx}")

            muscle_list = name_filter(muscle_list, 'tif')
            file_names = pd.DataFrame({"2.근육" : muscle_list},
                                        index = list(map(export_num, muscle_list))
                                       ).sort_index()
            file_names["2.근육"] = PATH + f"2.근육/tif/{person_idx}/" +\
                                    file_names["2.근육"]


            muscle_min = file_names.index.min()
            muscle_max = file_names.index.max()

            # 내장지방, 피하지방 추가하기
            for name in ['3.내장지방', '4.피하지방']:
                part_list = os.listdir(PATH + f"{name}/tif/{person_idx}")
                part_list = name_filter(part_list, 'tif')
                part_paths = pd.DataFrame( {name : part_list},
                                            index = list(map(export_num, part_list))
                                           ).sort_index()
                file_names[name] = part_paths.loc[muscle_min:muscle_max+1]
                file_names[name] = PATH + f"{name}/tif/{person_idx}/" +\
                                    file_names[name]

            original_path.index = original_max - original_path.index
            file_names["1.원본"] = original_path.loc[muscle_max:muscle_min-1]
            file_names["1.원본"] = PATH + f"1.원본/{person_idx}/" +\
                                    file_names["1.원본"]

            file_names["person_idx"] = person_idx
            file_names = file_names.reset_index()
            file_names = file_names.rename({'index':'loc_index'}, axis = 1)
            file_names = file_names.set_index(['person_idx', 'loc_index'])
            file_names = file_names[['1.원본', '2.근육','3.내장지방','4.피하지방']]
        except:
            print(person_idx)
            continue
        images_paths.append(file_names)

    images_paths =  pd.concat(images_paths)
    images_paths = images_paths.reset_index()
    return images_paths

def export_num(string):
    find_num = re.compile('[0-9.]+')
    number = find_num.findall(string[-10:-4])
    return int(number[0])

def name_filter(name_list, element):
    new_name_list = []
    for name in name_list:
        if element in name:
            new_name_list.append(name)
    return new_name_list

def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


def cal_epoch_score(metrics, metric_idx, step_size, run = "train"):
    # 가장 마지막 epoch의 step별 score 평균
    metrics_name = ['jaccard', 'dice', 'tpf', 'ftp']
    score_array = np.zeros([len(metrics[metric_idx].score_list[- step_size:]), 4])
    for row_idx, score_dic in enumerate(metrics[metric_idx].score_list[- step_size:]):
        score_array[row_idx, :] = np.fromiter(score_dic.values(), dtype=float)
    epoch_score = score_array.mean(axis = 0)
    score_dic_mean = {}
    for idx, part in enumerate(['muscle', 'visceral', 'subcutaneous', 'background']):
        score_dic_mean[f'{run}-{metrics_name[metric_idx]}-{part}'] = epoch_score[idx]
    return score_dic_mean