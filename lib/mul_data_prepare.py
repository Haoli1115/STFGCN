import torch
import numpy as np
import os
from lib.utils import StandardScaler
import yaml



# ! X shape: (B, T, N, C)
def spilit_data(data, split_train=0.6, split_test=0.2, his_len=12, pred_len=12,x_back_length=12):
    data = data
    data_len, num_of_vertices, features_d = data.shape
    len_per_day = 288
    split_train = int(split_train * data_len)
    split_test = int(split_test * data_len)
    long_his = x_back_length
    begin = 0
    end = data_len - long_his - pred_len - len_per_day
    for_len = end - begin

    x = np.zeros([for_len, his_len, num_of_vertices, features_d])
    y = np.zeros([for_len, pred_len, num_of_vertices, 1])
    x_backday = np.zeros([for_len, long_his, num_of_vertices, features_d])
    for i in range(begin, end):
        x_backday[i] = data[i:i + long_his, :, :]
        x[i] = data[i + len_per_day - his_len:i + len_per_day, :, :]
        y[i] = data[i + len_per_day: i + len_per_day + pred_len, :, 0:1]

    x_train = x[:split_train, ...]
    y_train = y[:split_train, ...]
    x_backday_train = x_backday[:split_train, ...]

    x_test = x[split_train:split_train + split_test, ...]
    y_test = y[split_train:split_train + split_test, ...]
    x_backday_test = x_backday[split_train:split_train + split_test, ...]

    x_val = x[split_train + split_test:data_len, ...]
    x_backday_val = x_backday[split_train + split_test:data_len, ...]
    y_val = y[split_train + split_test:data_len, ...]

    return x_train, x_backday_train, y_train, x_test, x_backday_test, y_test, x_val, x_backday_val, y_val


def get_dataloaders_from_index_data(
        filename, tod=False, dow=False, dom=False, batch_size=64, log=None,x_back_length=12,
):
    data = np.load(filename)["data"]
    # data = data[:2000]
    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    # if dom:
    #     features.append(3)
    data = data[..., features]

    x_train, x_backday_train, y_train, x_test, x_backday_test, y_test, x_val, x_backday_val, y_val = spilit_data(data,x_back_length=x_back_length)
    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_backday_train[..., 0] = scaler.transform(x_backday_train[..., 0])

    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_backday_val[..., 0] = scaler.transform(x_backday_val[..., 0])

    x_test[..., 0] = scaler.transform(x_test[..., 0])
    x_backday_test[..., 0] = scaler.transform(x_backday_test[..., 0])

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(x_backday_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(x_backday_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(x_backday_test), torch.FloatTensor(y_test)
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

    return trainset_loader, testset_loader, valset_loader,scaler


if __name__ == "__main__":
    with open("../model/STAEformer.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg["PEMS08"]

    filename = "../data/PEMS08/data.npz"
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        filename,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
    )
