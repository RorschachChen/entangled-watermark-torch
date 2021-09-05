import numpy as np
import os
import torch
from torch.utils.data import TensorDataset


def build(image_set, args):
    if image_set == 'train':
        x_train = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_train.npy')), 1, 2)
        y_train = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_train.npy'))
        tensor_x = torch.from_numpy(x_train)
        tensor_y = torch.from_numpy(y_train)
    else:
        x_test = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_test.npy')), 1, 2)
        y_test = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_test.npy'))
        tensor_x = torch.from_numpy(x_test)
        tensor_y = torch.from_numpy(y_test)
    return TensorDataset(tensor_x, tensor_y)
