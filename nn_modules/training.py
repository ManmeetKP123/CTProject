from distutils.command.config import config
import torch 
from torch import nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
from unet3d import UNet3D
from dataloaders import CTDataset
from config_file import config_file

print(torch.cuda.is_available())
print(torch.__version__)
#define the dice loss 
device = torch.device("cpu")
def dice_loss(self, y_pred, y_true, smooth = 1):

    #difference between flatten vs view??

    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    intersection = (y_true * y_pred).sum()
    dice = (2 * intersection  + smooth)/(y_true.sum() + y_pred.sum() + smooth)

    return 1 - dice()

img_dir = "/ubc/ece/home/ra/other/manmeetp/CTProject/resampled_input_data"
annotations = "/ubc/ece/home/ra/other/manmeetp/CTProject/MED_ABD_LYMPH_MASKS"
trainset = CTDataset(annotations_folder=annotations, img_dir=img_dir)
train_loader = DataLoader(trainset, batch_size = config_file.batch_size, shuffle = True)

model = UNet3D()
model.to(device)
criterion = dice_loss

#Paper used optimizer 
optimizer = torch.optim.Adam(model.parameters(), config_file.learning_rate)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())



# for epoch in range(config_file.epochs):
#     scheduler.step(epoch)
#     model.train()

#     for batch_idx, (x, y) in enumerate(train_loader):
#         x, y = x.float().to(device), y.float().to(device)
#         pred = model(x)
#         loss = criterion(pred, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

