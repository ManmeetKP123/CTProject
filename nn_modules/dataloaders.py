import torch 
from torch.utils.data import Dataset 
import os

class CTDataset(Dataset):
    def __init__(self, annotations_folder, img_dir, transform=None, target_transform=None) -> None:
        self.list_files = os.listdir(img_dir)
        self.labels = os.listdir(annotations_folder)

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, idx):
        volume = self.img_dir[0]
