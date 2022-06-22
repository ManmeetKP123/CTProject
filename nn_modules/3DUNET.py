import torch 
import torch.nn as nn 
import os
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms 

class NeuralNetwork:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    