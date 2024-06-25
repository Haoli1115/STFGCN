import numpy as np
import torch
import pickle
import random
import os
import json
import csv
from scipy.sparse.linalg import eigs


class StandardScaler:
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)


def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data






class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f"Shape: {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)


def vrange(starts, stops):
    """Create ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)
        
        Lengths of each range should be equal.

    Returns:
        numpy.ndarray: 2d array for each range
        
    For example:

        >>> starts = [1, 2, 3, 4]
        >>> stops  = [4, 5, 6, 7]
        #>>> vrange(starts, stops)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6]])

    Ref: https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
    """
    stops = np.asarray(stops)
    l = stops - starts  # Lengths of each range. Should be equal, e.g. [12, 12, 12, ...]
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])


def print_model_params(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("%-40s\t%-30s\t%-30s" % (name, list(param.shape), param.numel()))
            param_count += param.numel()
    print("%-40s\t%-30s" % ("Total trainable params", param_count))


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1

    return A


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


