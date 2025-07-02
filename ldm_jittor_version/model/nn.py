import jittor as jt
import jittor.nn as nn

jt.flags.use_cuda = 1

def mean_flat(tensor: jt.Var):
    """
    Jittor Implementation of mean_flat
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


