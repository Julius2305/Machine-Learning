import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from tqdm import tqdm


def crop_tensor(x, cropping_size):
    if((x.size()[2]/2) >= cropping_size):
        return x[0:, 0:, cropping_size:-cropping_size, cropping_size:-cropping_size]

if __name__ == '__main__':
    list = [[[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]]]

    tensor = torch.tensor(list)

    tensor = crop_tensor(tensor, 1)
    print(tensor)
