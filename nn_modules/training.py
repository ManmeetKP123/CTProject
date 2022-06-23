import torch 
from torch import nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
import unet3d

#define the dice loss 
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average = True) -> None:
        super(DiceLoss, self).__init__()
    
    def forward(self, y_pred, y_true, smooth = 1):

        #difference between flatten vs view??

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_true * y_pred).sum()
        dice = (2 * intersection  + smooth)/(y_true.sum() + y_pred.sum() + smooth)

        return 1 - dice()