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


def myimshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image, cmap='gray')
    ax.axis('off')
    return h

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


model = UDnCNN(6).to(device)
lr = 1e-4
adam = torch.optim.Adam(model.parameters(), lr=lr)

model, adam, _ = utils.load_ckpt('gaussian_unet_200_grayscale.pth', model, adam)


# datasets to test upon
test_ds = NoisyBSDSDataset('BSDS300/images/test', image_size=(200, 200))
idx = 10
for i, name in enumerate(os.listdir('BSDS300/images/test')):
    if name == 'spiral.jpg':
        idx = i
        break
        
noisy, clean = test_ds[idx]
noisy = noisy.unsqueeze(0).to(device=device)
# clean = clean.unsqueeze(0).to(device=device)

_, axs = plt.subplots(ncols=3, figsize=(16,16), sharex='all', sharey='all')
images = []
titles = ['Noisy', 'Clean', 'Difference']
images.append(noisy[0])


model.eval()
with torch.no_grad():
    y = model(noisy)

images.append(y[0])
images.append(2*(y[0]-noisy[0]))

for i in range(len(images)):
    myimshow(images[i], ax=axs[i])
    axs[i].set_title(f'{titles[i]}')

plt.show()



