from typing import Iterable
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.func as func
from src.meta_alg_base import MetaLearningAlgBase


class KronPrecond(nn.Module):
    """Kronecker-factorized Preconditioner
    """
    def __init__(self, named_params: Iterable[tuple[str, torch.Tensor]], **kwargs) -> None:
        super().__init__()
        self.mc_weight = nn.ParameterDict()

        for name, param in named_params:
            name = name.replace('.', '_')
            param_size = param.size()
            param_dim = param.dim()

            if param_dim == 1:      # Conv2d().bias / Linear().bias
                self.mc_weight[name] = nn.Parameter(torch.ones_like(param, **kwargs))
            else:                   # Linear().weight / Conv2d().weight
                self.mc_weight[name+'_mc0'] = nn.Parameter(torch.eye(param_size[0], **kwargs))
                self.mc_weight[name+'_mc1'] = nn.Parameter(torch.eye(param_size[1], **kwargs))
                if param_dim == 4:  # Conv2d().weight
                    self.mc_weight[name+'_mc2'] = nn.Parameter(torch.eye(param_size[2] * param_size[3], **kwargs))

    def forward(self, named_grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pointer = 0
        precond_grads = dict()

        for name, grad in named_grads.items():
            name_mc = name.replace('.', '_')
            param_dim = grad.dim()

            if param_dim == 1:  # Conv2d().bias / Linear().bias
                precond_grad = self.mc_weight[name_mc] * grad
                pointer += 1
            elif param_dim == 2:  # Linear().weight
                precond_grad = self.mc_weight[name_mc+'_mc0'] @ grad @ self.mc_weight[name_mc+'_mc1']
                pointer += 2
            elif param_dim == 4:  # Conv2d().weight
                precond_grad = torch.einsum('ijk,il->ljk',
                                           grad.flatten(start_dim=2),
                                           self.mc_weight[name_mc+'_mc0'])
                precond_grad = self.mc_weight[name_mc+'_mc1'] @ precond_grad @ self.mc_weight[name_mc+'_mc2']
                pointer += 3
            else:
                raise NotImplementedError

            precond_grads[name] = precond_grad.view_as(grad)

        return precond_grads


class MetaCurvature(MetaLearningAlgBase):
    """Meta-Curvature
    https://proceedings.neurips.cc/paper_files/paper/2019/hash/57c0531e13f40b91b3b0f1a30b529a1d-Abstract.html
    """
    def __init__(self, args) -> None:
        super().__init__(args)

    def _get_meta_model(self, **kwargs) -> dict[str, nn.Module]:
        return {'init': self._get_base_model(**kwargs),
                'precond': KronPrecond(self.base_model.named_parameters(), **kwargs)}

    def adapt(self, trn_inputs: torch.Tensor, trn_targets: torch.Tensor,
              first_order: bool = False) -> OrderedDict[str, nn.Parameter]:
        batch_size = trn_inputs.size(0)
        batch_params = OrderedDict()
        for name, param in self.meta_model['init'].named_parameters():
            batch_params[name] = param.expand(batch_size, *param.size())

        for _ in range(self.args.task_iter):
            batch_grads = self.batch_grad_fn(batch_params, trn_inputs, trn_targets)
            batch_grads = func.vmap(self.meta_model['precond'])(batch_grads)

            for name, param in batch_params.items():
                batch_params[name] = param - self.args.task_lr * batch_grads[name]

        return batch_params
