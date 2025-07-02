import math

import torch
import torch.nn as nn

def mean_flat(tensor: torch.Tensor):
    """
    在所有维度上做平均（除了批次维度）
    :param tensor:
    :return:
    """

    return tensor.mean(dim=list(range(1, len(tensor.shape))))



