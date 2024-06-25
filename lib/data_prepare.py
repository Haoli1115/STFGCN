import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange


# ! X shape: (B, T, N, C)
def spilit_data(data, split_train=0.6, split_test=0.2, his_len=12, pred_len=12):
    data = data
    data_len, num_of_vertices, features_d = data.shape
    len_per_day = 288
    split_train = int(split_train * data_len)
    split_test = int(split_test * data_len)
    for_len = data_len - pred_len - his_len

    x = np.zeros([for_len,his_len,num_of_vertices,features_d])
    y = np.zeros([for_len,pred_len,num_of_vertices,1])
    for i in range(0, data_len - pred_len - his_len):
        x[i] = data[i: i + his_len, :, :]
        y[i] = data[i + his_len: i + his_len + pred_len, :, 0:1]

    x_train = x[:split_train, ...]
    y_train = y[:split_train, ...]

    x_test = x[split_train:split_train + split_test, ...]
    y_test = y[split_train:split_train + split_test, ...]

    x_val = x[split_train + split_test:data_len, ...]
    y_val = y[split_train + split_test:data_len, ...]

    return x_train , y_train, x_test, y_test, x_val, y_val


def get_dataloaders_from_index_data(
        filename, tod=False, dow=False, dom=False, batch_size=64, log=None
):
    data = np.load(filename)["data"]
    #data = data[:2000]
    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    # if dom:
    #     features.append(3)
    data = data[..., features]


    x_train, y_train, x_test, y_test, x_val, y_val = spilit_data(data)
    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])


    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler
