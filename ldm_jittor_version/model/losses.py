import numpy as np
import jittor as jt


def normal_kl(mean1, logvar1, mean2, logvar2):

    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, jt.Var):
            tensor = obj
            break
    assert tensor is not None, "at least 1 argument must be a tensor"

    logvar1, logvar2 = [
        x if isinstance(x, jt.Var) else jt.Var(x)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        +logvar2
        -logvar1
        +jt.exp(logvar1-logvar2)
        +((mean1 - mean2) ** 2) * jt.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):

    return 0.5 * (1.0 + jt.tanh(np.sqrt(2.0 / np.pi) * (x+0.044715+jt.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    pass

def vae_loss(recon_x, x, mu, logvar, kld_weight=0.001):
    pass



