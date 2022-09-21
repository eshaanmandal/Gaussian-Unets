import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


def imshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    ax.show()
    return h


def save_ckpt(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)

def load_ckpt(checkpoint_path, model, optimizer):
    checkpoint =  torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']