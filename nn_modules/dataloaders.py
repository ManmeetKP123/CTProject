import torch 
from torch.utils.data import Dataset 
import os
import random
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import torchio as tio

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
        volume_path = os.path.join(self.img_dir, self.list_files[idx])
        volume = nib.load(volume_path).get_data()

        #do the same for masks
        mask_path = os.path.join(self.annotations_folder, self.labels[idx])
        mask = nib.load(mask_path).get_data()

        volume, mask = self.dimension_adjust(volume, mask)
        volume, mask = self.preprocessing(volume, mask)

        return torch.tensor(volume), torch.tensor(mask)
    


    def preprocessing(self, image, mask):
        #normalization
        # transform = tio.transforms.RescaleIntensity(out_min_max: Union[float, Tuple[float, float]] = (0, 1), 
        # percentiles: Union[float, Tuple[float, float]] = (0, 100), 
        # masking_method: Optional[Union[str, Callable[[Tensor], Tensor], int, Tuple[int, int, int], 
        # Tuple[int, int, int, int, int, int]]] = None, in_min_max: Optional[Tuple[float, float]] = None, **kwargs)[source]
        return image, mask



    def dimension_adjust(self, image, mask):

        if np.shape(image) != np.shape(mask):
            raise Exception("Image and Annotation are not of the same shape")
        #in the case that the actual image is just smaller than 128 slices
        if (np.shape(mask)[2] < 128):
            #do something
            curr_depth = np.shape(mask)[2]

            """
            pad: a list of length 2 * len(source.shape) of the form (begin last axis, end last axis, 
            begin 2nd to last axis, end 2nd to last axis, begin 3rd to last axis, etc.) that states 
            how many dimensions should be added to the beginning and end of each axis,
            """

            #just padding the last dimension so only the depth
            pad = 128 - curr_depth
            mask = F.pad(input=mask, pad=(0, pad), mode='constant', value=0)
            image = F.pad(input = image, pad=(0, pad), mode='constant', value=0)

        else:
            #finding the peak of the annotation instead of randomly cropping the last dimension
            peak_sum = np.sum(image, axis=(0, 1))
            peak_loc = np.argmax(peak_sum)

            start = peak_loc
            end = peak_loc

            no_slices = 0
            #gets the indices of the 128 slices around the peak 
            while True:
                start -= 1
                no_slices += 1
                if (no_slices == 128):
                    break
                end += 1
                no_slices += 1
                if (no_slices == 128):
                    break
            image = image[:, :, start:end]
            mask = mask[:, :, start:end]

        return image, mask

