import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
import time
from dataset import NoisyBSDSDataset
from models import UDnCNN
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def train(epochs, train_dl, val_dl, model, loss_fn, optimizer, device='cpu'):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (X,y) in enumerate(train_dl):
            X = X.to(device)
            y = y.to(device)

            model.train()
            forward_pass = model(X)
            loss  = loss_fn(forward_pass, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            val_loss = 0.0

            for val_X, val_y in val_dl:
                val_X = val_X.to(device)
                val_y = val_y.to(device)
                model.eval()
                with torch.no_grad():
                    fp = model(val_X)
                    val_loss += loss_fn(fp, val_y).item()


        print(f"Epoch {epoch+1} | Train Loss is : {running_loss / len(train_dl)} | Validation Loss is {val_loss/len(val_dl)}")



# dataloaders
train_ds = NoisyBSDSDataset('BSDS300/images/train')
validation_ds = NoisyBSDSDataset('BSDS300/images/validation')

train_dl = td.DataLoader(train_ds, 4, True, num_workers=0)
val_dl = td.DataLoader(validation_ds, 4, True, num_workers=0)

model = UDnCNN(6).to(device=device)
lr = 1e-4
adam = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()


# train
# train(100, train_dl, val_dl, model, loss_fn, adam, device=device)

# checkpoint = {
#     'state_dict': model.state_dict(),
#     'epoch' : 100,
#     'optimizer': adam.state_dict(),
# }

# utils.save_ckpt(checkpoint, 'gaussian_unet_100_grayscale.pth')


# continue training

model, adam, _ = utils.load_ckpt('gaussian_unet_100_grayscale.pth', model, adam)
train(100, train_dl, val_dl, model, loss_fn, adam, device=device)
checkpoint = {
    'state_dict': model.state_dict(),
    'epoch' : 200,
    'optimizer': adam.state_dict(),
}

utils.save_ckpt(checkpoint, 'gaussian_unet_200_grayscale.pth')