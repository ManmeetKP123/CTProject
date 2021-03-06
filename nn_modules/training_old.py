from distutils.command.config import config
import torch 
from torch import nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
from nn_modules.dataloaders import CTDataset
from config_file import config_file
from nn_modules.models.unet_from_scratch import UNet

print(torch.cuda.is_available())
print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#define the dice loss 
def dice_loss(y_pred, y_true, smooth = 1):

    #difference between flatten vs view??

    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    print("dimension of prediction " + str(y_pred.size()))
    print("dimension of true " + str(y_true.size()))
    smooth = 1
    intersection = (y_true * y_pred).sum()
    dice = (2 * intersection  + smooth)/(y_true.sum() + y_pred.sum() + smooth)

    return 1 - dice()

img_dir = "/ubc/ece/home/ra/other/manmeetp/CTProject/training_set/128_image_arrays"
annotations = "/ubc/ece/home/ra/other/manmeetp/CTProject/training_set/128_mask_arrays"
trainset = CTDataset(annotations_folder=annotations, img_dir=img_dir)
train_loader = DataLoader(trainset, batch_size = config_file.batch_size, shuffle = True)

model = UNet()
model.to(device)
criterion = dice_loss

#Paper used optimizer 
optimizer = torch.optim.Adam(model.parameters(), config_file.learning_rate)
lambda1 = lambda epoch: 0.65 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())



for epoch in range(config_file.epochs):
    scheduler.step()
    model.train()

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.float().to(device), y.float().to(device)
        pred = model(x)
        print("this type here " + str(type(y)))
        print("this dimension " + str(y.size()))
        loss = criterion(pred, y)
        print("Dice loss " + str(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

