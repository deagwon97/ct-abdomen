import numpy as np

import torch

import segmentation_models_pytorch as smp


def cal_jaccard(real_mask, predict_mask, eps = 1e-8):
    real_mask = real_mask.argmax(axis = 1).cpu()
    predict_mask = predict_mask.argmax(axis = 1).cpu()
    jaccard_scores = {}
    for idx, part in enumerate(['muscle', 'visceral', 'subcutaneous', 'background']):
        predict_part = (predict_mask == idx)
        real_part = (real_mask == idx)
        union = predict_part.logical_or(real_part)
        intersection =  predict_part.logical_and(real_part)
        jaccard_scores[part] = intersection.sum() / (union.sum() + eps)
    return jaccard_scores

def cal_dice(real_mask, predict_mask):
    real_mask = real_mask.argmax(axis = 1).cpu()
    predict_mask = predict_mask.argmax(axis = 1).cpu()
    dice_scores = {}
    for idx, part in enumerate(['muscle', 'visceral', 'subcutaneous', 'background']):
        predict_part = (predict_mask == idx)
        real_part = (real_mask == idx)
        (predict_part.sum() + real_part.sum())
        intersection =  predict_part.logical_and(real_part)
        dice_scores[part] = intersection.sum() * 2 / (predict_part.sum() + real_part.sum() + 1e-8)
    return dice_scores

def cal_tpf(real_mask, predict_mask):
    real_mask = real_mask.argmax(axis = 1).cpu()
    predict_mask = predict_mask.argmax(axis = 1).cpu()
    tpf_scores = {}
    for idx, part in enumerate(['muscle', 'visceral', 'subcutaneous', 'background']):
        predict_part = (predict_mask == idx)
        real_part = (real_mask == idx)
        intersection =  predict_part.logical_and(real_part)
        tpf_scores[part] = intersection.sum() / (real_part.sum() + 1e-8)
    return tpf_scores

def cal_fpf(real_mask, predict_mask):
    real_mask = real_mask.argmax(axis = 1).cpu()
    predict_mask = predict_mask.argmax(axis = 1).cpu()
    fpf_scores = {}
    for idx, part in enumerate(['muscle', 'visceral', 'subcutaneous', 'background']):
        predict_part = (predict_mask == idx)
        real_part = (real_mask == idx)
        intersection = predict_part.logical_and(real_part)
        FP = predict_part.logical_xor(intersection)
        all_area = predict_part.size(0) * predict_part.size(1) * predict_part.size(2)
        fpf_scores[part] = FP.sum() / (all_area - real_part.sum() + 1e-8)
    return fpf_scores


class Multi_Scores(smp.utils.metrics.IoU):
    __name__ = 'multi_scores'
    def __init__(self, eps=1e-7, metric = None, threshold=0.5, name = None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = smp.base.modules.Activation('softmax')
        self.metric = metric
        self.__name__ = name
        self.score_list = []
        
    def forward(self, y_pr, y_gt):
        scores = self.metric(y_gt, y_pr)
        self.score_list.append(scores)
        vals = np.fromiter(scores.values(), dtype=float)
        return torch.tensor(vals).mean()
    
    