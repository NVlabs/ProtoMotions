import torch.nn as nn
from torch.nn import functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_activation_func(activation_name, return_type="nn"):
    if activation_name.lower() == "tanh":
        activation = (nn.Tanh(), F.tanh)
    elif activation_name.lower() == "relu":
        activation = (nn.ReLU(), F.relu)
    elif activation_name.lower() == "elu":
        activation = (nn.ELU(), F.elu)
    elif activation_name.lower() == "gelu":
        activation = (nn.GELU(), F.gelu)
    elif activation_name.lower() == "identity":
        activation = (nn.Identity(), lambda x: x)
    elif activation_name.lower() == "silu":
        activation = (nn.SiLU(), F.silu)
    elif activation_name.lower() == "mish":
        activation = (nn.Mish(), F.mish)
    else:
        raise NotImplementedError(
            "Activation func {} not defined".format(activation_name)
        )

    if return_type == "nn":
        return activation[0]
    elif return_type == "functional":
        return activation[1]
    else:
        raise NotImplementedError("Return type {} not defined".format(return_type))
