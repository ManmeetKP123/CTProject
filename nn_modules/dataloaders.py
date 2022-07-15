import torch 
from torch.utils.data import Dataset 
import os
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import torchio as tio
import albumentations as A

from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms


class CTDataset(Dataset):
    def __init__(self, annotations_folder, img_dir, transform=None, target_transform=None) -> None:
        self.img_dir = img_dir
        self.annotations_folder = annotations_folder

        self.list_files = os.listdir(img_dir)
        self.labels = os.listdir(annotations_folder)

    def __len__(self):
        return len(self.list_files)                     
    
    def __getitem__(self, idx):
        #all the data augumentation and stuff must be done in this method cuz this is what is being 
        #fed into the network 

        #get the volume in numpy array format
        #EXPECTS A SPATIAL RESOLUTION OF 128 X 128 X depth
        volume_path = os.path.join(self.img_dir, self.list_files[idx])
        volume = np.load(volume_path)

        #do the same for masks
        mask_path = os.path.join(self.annotations_folder, self.labels[idx])
        mask = np.load(mask_path)

        #normalization
        volume, mask = self.normalizing(volume, mask)
        volumeTensor, maskTensor = torch.tensor(volume), torch.tensor(mask)
        volumeTensor, maskTensor = self.augumentation(volumeTensor, maskTensor)
        volumeTensor = volumeTensor.reshape(1, volumeTensor.size(dim=0), volumeTensor.size(dim=1), volumeTensor.size(dim=2))
        #maskTensor = maskTensor.reshape(1, maskTensor.size(dim=0), maskTensor.size(dim=1), maskTensor.size(dim=2))

        return volumeTensor, maskTensor


    def normalizing(self, volume, mask):
        volume_norm = []
        mask_norm = []
        for slice_index in range(0, np.shape(volume)[2]):
            vol_slice = volume[:, :, slice_index]
            norm_slice = (vol_slice - np.min(vol_slice)) / (np.max(vol_slice) - np.min(vol_slice))
            volume_norm.append(norm_slice)

        for slice_index in range(0, np.shape(mask)[2]):
            mask_slice = mask[:, :, slice_index]
            norm_slice = (mask - np.min(mask_slice)) / (np.max(mask_slice) - np.min(mask_slice))
            mask_norm.append(norm_slice)
        print("success")
        return np.array(volume_norm), np.array(mask_norm)

    """
    Data Augumentation code:
    1. Horizontal and Vertical Flipping 
    2. Random rotation in the range [-20*, 20*]
    3. Translation upto 10% of the axis dimension
    4. Zoom between [80, 120]% of the axial plane 

    Assumption: First dimension is the number of rows, then columms, then depth at the end.  
    Albumentations requires a newer version of glibc so use PyTorch  

    PyTorch expects Tensors     
    """

    #took away the affine code bc it was causing errors 
    def augumentation(self, imageTensor, maskTensor):
        p = 0.5
        transform = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(p), 
            transforms.RandomVerticalFlip(p),   
        )

        imageTensor = transform(imageTensor)
        maskTensor = transform(maskTensor)

        return imageTensor, maskTensor

