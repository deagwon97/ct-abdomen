import os
import sys
sys.path.append("../src/")
import wandb
import random
import json
import subprocess
import numpy as np
import tqdm
import albumentations as albu
import matplotlib.pyplot as plt
import torch
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold

#custom modules
from datasets.datasets import CT_Dataset
from utils.utils import cal_epoch_score, figure_to_array
from utils.report import make_report
from metrix.metrix import Multi_Scores, cal_jaccard, cal_dice, cal_tpf, cal_fpf

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.
    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  
    

# wandb 설정
if __name__ == "__main__":
    # input path
    #if sys.argv[1] == '-config':
    #    config_json = sys.argv[2]
    
    config_json = "config.json"
    #with open('../notebooks/config.json') as json_file:
    with open(config_json) as json_file_path:
        CONFIG = json.load(json_file_path)
        
    CONFIG['augmentation']= [
                                   albu.Transpose(p=0.5),
                                   albu.RandomRotate90(3),
                                   albu.Rotate(p=1, border_mode = 1),
                                   albu.GridDistortion(p = 0.2), # 2배
                                   albu.Cutout(p=0.2, num_holes=30, max_h_size=20, max_w_size=20),
                               ]
    CONFIG['num_epochs']= 40
    CONFIG['batch_size']= 10
    CONFIG['optimizer']= torch.optim.Adam
    CONFIG['scheduler']= lr_scheduler.CosineAnnealingWarmRestarts
    CONFIG['loss']= smp.utils.losses.DiceLoss
    CONFIG['device']= "cuda:1"
    #wandb.init(project="ct_segment", config=CONFIG,  reinit = True)
    #wandb.run.name = CONFIG['name']
    #wandb.run.save()
    seed_everything(CONFIG['seed'])
    
    images_paths = os.listdir(CONFIG['path']+'/img')
    images_paths_onlypng = []
    for file_name in images_paths:
        if '.png' in file_name:
            images_paths_onlypng.append(file_name)
    images_paths = images_paths_onlypng
    #images_paths.pop(-1)
    images_paths = np.array(images_paths)
    train_images = images_paths[len(images_paths) // 10 : ]
    test_images  = images_paths[:len(images_paths) // 10]
    kfold = KFold(n_splits=5, shuffle = False)
    target_fold_index = 1
    train_metrics_list = []
    valid_metrics_list = []

    def get_training_augmentation():
        transform = CONFIG['augmentation']
        return albu.Compose(transform)
    
    for fold_index, (trn_idx, val_idx) in enumerate(kfold.split(train_images)):
        if fold_index == target_fold_index:
            print(fold_index)
            train_fold = train_images[trn_idx]
            valid_fold = train_images[val_idx]
            ENCODER = CONFIG['model']
            ENCODER_WEIGHTS = CONFIG['pretrain']
            ACTIVATION = 'softmax'
            CLASSES = 4

            model = smp.Unet(
                encoder_name=ENCODER, 
                encoder_weights=ENCODER_WEIGHTS, 
                in_channels = 1,
                classes=CLASSES, 
                activation = ACTIVATION,
            )

            #preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


            train_dataset = CT_Dataset(
                path = CONFIG['path'],
                imglist = train_fold,
                augmentation = get_training_augmentation(), 
                preprocessing = None,#get_preprocessing(preprocessing_fn),
                classes=4,
            )

            valid_dataset = CT_Dataset(
                path = CONFIG['path'],
                imglist = valid_fold,
                augmentation = None,#get_training_augmentation(), 
                preprocessing = None,#get_preprocessing(preprocessing_fn),
                classes=4,
            )


            BATCH_SIZE = CONFIG['batch_size']
            train_step_size = train_dataset.__len__() // BATCH_SIZE
            valid_step_size = valid_dataset.__len__() // BATCH_SIZE
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


            #loss = smp.utils.losses.DiceLoss()
            #loss = utils.train.WeightedDiceLoss()
            loss = CONFIG['loss']()

            train_metrics = [
                Multi_Scores(metric = cal_jaccard, name      = 'avg_jaccard'),
                Multi_Scores(metric = cal_dice,    name      =    'avg_dice'),
                Multi_Scores(metric = cal_tpf,     name      =     'avg_tpf'),
                Multi_Scores(metric = cal_fpf,     name      =     'avg_fpf')
            ]
            valid_metrics = [
                Multi_Scores(metric = cal_jaccard, name      = 'avg_jaccard'),
                Multi_Scores(metric = cal_dice,    name      =    'avg_dice'),
                Multi_Scores(metric = cal_tpf,     name      =     'avg_tpf'),
                Multi_Scores(metric = cal_fpf,     name      =     'avg_fpf')
            ]

            optimizer = CONFIG['optimizer']([ 
                dict(params=model.parameters(), lr=0.0001),
            ])

            scheduler = CONFIG['scheduler'](optimizer, 10, 2, eta_min=1e-6)
            train_epoch = smp.utils.train.TrainEpoch(
                model, 
                loss=loss, 
                metrics=train_metrics, 
                optimizer=optimizer,
                device=CONFIG['device'],
                verbose=True,
            )

            valid_epoch = smp.utils.train.ValidEpoch(
                model, 
                loss=loss, 
                metrics=valid_metrics, 
                device=CONFIG['device'],
                verbose=True,
            )

            # 한 폴드당 150 Epoch 수행
            NUM_EPOCH = CONFIG['num_epochs']
            min_score = np.Inf
            if not os.path.exists(CONFIG['model_path']):
                subprocess.call(["mkdir",
                             CONFIG['model_path']],
                             shell = False)
            #MODEL = train_name
            patient = 0
            max_valid_dice = 0
            for i in range(0, NUM_EPOCH):

                print('\nEpoch: {}'.format(i))
                train_logs = train_epoch.run(train_loader)
                valid_logs = valid_epoch.run(valid_loader)


                score_log = {}

                train_log = {}
                for metric_idx, cal in enumerate(['jaccard', 'dice',
                                                  'tpf', 'ftp']):
                    score_dic = cal_epoch_score(train_metrics,
                                                metric_idx,
                                                train_step_size,
                                                run = 'train')
                    train_log.update(score_dic)
                valid_log = {}
                for metric_idx, cal in enumerate(['jaccard', 'dice',
                                                  'tpf', 'ftp']):
                    score_dic = cal_epoch_score(valid_metrics,
                                                metric_idx,
                                                valid_step_size,
                                                run = 'valid')
                    valid_log.update(score_dic)

                score_log.update(train_log)
                score_log.update(valid_log)
                #wandb.log(score_log)

                
                scheduler.step()
                if max_valid_dice < valid_logs['avg_dice']:
                    max_valid_dice = valid_logs['avg_dice']
                    torch.save(model, f"{CONFIG['model_path']}/{CONFIG['model']}_{max_valid_dice:.4f}loss_{i}epochs.pth")


                    make_report(train_log).to_csv(f"{CONFIG['model_path']}/train_report.csv")
                    make_report(valid_log).to_csv(f"{CONFIG['model_path']}/valid_report.csv")

                    with torch.no_grad():
                        sample_img, sample_mask = valid_dataset[np.random.randint(valid_dataset.__len__())]
                        sample_tensor = torch.cuda.FloatTensor(sample_img[np.newaxis,...])
                        sample_pred   = model.predict(sample_tensor).cpu().numpy()[0]

                        plt.figure(figsize=(20,5))
                        #img, mask = dataset[img_id]
                        plt.subplot(1,4,1)
                        plt.imshow(sample_img[0,:,:], cmap='bone') # 원본 # permute는 축 변경
                        plt.gca().set_title("Image")
                        plt.subplot(1,4,2)
                        plt.imshow(sample_img[0,:,:], cmap='bone') # 원본 # permute는 축 변경
                        plt.imshow(sample_mask.transpose([1,2,0]).argmax(axis = 2), alpha=0.3, cmap='flag') # 레이블
                        plt.gca().set_title("True Mask")
                        plt.subplot(1,4,3)
                        plt.imshow(sample_img[0,:,:], cmap='bone') # 원본 # permute는 축 변경
                        plt.imshow(sample_pred.transpose([1,2,0]).argmax(axis = 2), alpha=0.3, cmap='flag') # 레이블
                        plt.gca().set_title("Predicted Mask")
                        plt.subplot(1,4,4)
                        plt.imshow(sample_mask.transpose([1,2,0]).argmax(axis = 2), alpha=0.3, cmap='flag') # 레이블
                        plt.imshow(sample_pred.transpose([1,2,0]).argmax(axis = 2), alpha=0.3, cmap='flag') # 레이블
                        plt.gca().set_title("Overlay")

                        # save in wandb
                        plt.tight_layout()
                        fig = plt.gcf()
                        figure_array = figure_to_array(fig)
                        '''
                        wandb.log({"examples": [wandb.Image(figure_array,
                                                            caption=f"{CONFIG['name']}_{max_valid_dice:.4f}loss_{i}epochs")]})
                        '''
                        plt.show()

                else:
                    patient += 1
                    if patient > 5:
                        print("early stopping")
                        break

            train_metrics_list.append(train_metrics)
            valid_metrics_list.append(valid_metrics)
            target_fold_index += 1

            