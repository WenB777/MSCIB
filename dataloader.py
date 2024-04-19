from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat('data/MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat('data/MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat('data/MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    dataset = MNIST_USPS('data/MNIST_USPS.mat')
    dims = [784, 784]
    view = 2
    class_num = 10
    data_size = 5000

    return dataset, dims, view, data_size, class_num
