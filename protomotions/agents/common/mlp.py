import torch
from torch import nn, Tensor
from hydra.utils import instantiate
from protomotions.agents.common.common import NormObsBase
from protomotions.utils import model_utils


def build_mlp(config, num_in: int, num_out: int):
    indim = num_in
    layers = []
    for i, layer in enumerate(config.layers):
        layers.append(nn.Linear(indim, layer.units))
        if layer.use_layer_norm and i == 0:
            layers.append(nn.LayerNorm(layer.units))
        layers.append(model_utils.get_activation_func(layer.activation))
        indim = layer.units

    layers.append(nn.Linear(indim, num_out))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__()
        self.config = config
        self.mlp = build_mlp(self.config, num_in, num_out)

    def forward(self, input_dict, *args, **kwargs):
        if isinstance(input_dict, torch.Tensor):
            return self.mlp(input_dict)
        return self.mlp(input_dict[self.config.obs_key])


class MLP_WithNorm(NormObsBase):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in, num_out)
        self.mlp = build_mlp(self.config, num_in, num_out)

    def forward(self, input_dict, return_norm_obs=False):
        obs = super().forward(input_dict[self.config.obs_key])
        outs: Tensor = self.mlp(obs)

        if return_norm_obs:
            return {"outs": outs, f"norm_{self.config.obs_key}": obs}
        else:
            return outs


class MultiHeadedMLP(nn.Module):
    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config
        self.num_out = num_out

        input_models = {}
        self.feature_size = 0
        for key, input_cfg in self.config.input_models.items():
            model = instantiate(input_cfg)
            input_models[key] = model
            self.feature_size += model.num_out
        self.input_models = nn.ModuleDict(input_models)

        self.trunk: MLP = instantiate(self.config.trunk, num_in=self.feature_size)

    def forward(self, input_dict, return_norm_obs=False):
        if return_norm_obs:
            norm_obs = {}
        outs = []

        for key, model in self.input_models.items():
            out = model(input_dict, return_norm_obs=return_norm_obs)
            if return_norm_obs:
                out, norm_obs[f"norm_{model.config.obs_key}"] = (
                    out["outs"],
                    out[f"norm_{model.config.obs_key}"],
                )
            outs.append(out)

        outs = torch.cat(outs, dim=-1)

        outs: Tensor = self.trunk(outs)

        if return_norm_obs:
            ret_dict = {**{"outs": outs}, **norm_obs}
            return ret_dict
        else:
            return outs
