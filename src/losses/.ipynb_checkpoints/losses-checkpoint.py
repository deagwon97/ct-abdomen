import segmentation_models_pytorch as smp


class WeightedDiceLoss(smp.utils.base.Loss):
    __name__ = 'weighted_dice_loss'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diceloss = smp.utils.losses.DiceLoss()
        
    def forward(self, y_pr, y_gt):
        weighted_dice_loss = self.diceloss.forward(y_pr[...,0,:,:], y_gt[...,0,:,:]) / 0.084 +\
                                self.diceloss.forward(y_pr[...,1,:,:], y_gt[...,1,:,:]) / 0.058 +\
                                self.diceloss.forward(y_pr[...,2,:,:], y_gt[...,2,:,:]) / 0.087 +\
                                self.diceloss.forward(y_pr[...,3,:,:], y_gt[...,3,:,:]) / 0.770
        
        return weighted_dice_loss / 100
