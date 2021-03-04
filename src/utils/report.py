import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_result(score_list, title = None, step_size = None):
    plt.figure(figsize = (12,3))
    score_array = np.array(score_list)
    plt.plot(score_array[::step_size,0], '-o', color = 'black',label  = 'Muscle')
    plt.plot(score_array[::step_size,1], '-x', color = 'black', label = 'Visceral')
    plt.plot(score_array[::step_size,2], '-^', color = 'black', label = 'Subcutaneous')
    plt.plot(score_array[::step_size,3], '--D', color = 'gray', label = 'Background')
    plt.title(title)
    plt.xlabel('epochs')
    plt.grid()
    plt.legend()
    plt.show()
    
def make_report(epoch_score_dic):
    epoch_score_array = np.fromiter(epoch_score_dic.values(), dtype=float).reshape([4,4])
    score_df = pd.DataFrame(epoch_score_array,
                columns = ['Muscle', 'Visceral', 'Subcutaneous', 'Background'],
                index = ['Jaccard score','Dice score', 'TPF', 'FPF'])
    return score_df

def cv_report(score_metrics_list):
    score_df = pd.DataFrame(np.zeros([4,4]), columns = ['Muscle', 'Visceral', 'Subcutaneous', 'Background'])
    for score_metrics in score_metrics_list:
        score_df += make_report(score_metrics)
    return score_df / len(score_metrics_list)