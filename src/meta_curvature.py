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
        self.kron_weights = nn.ParameterDict()

        for name, param in named_params:
            name = name.replace('.', '_')
            param_size = param.size()
            param_dim = param.dim()

            if param_dim == 1:      # Conv2d().bias / Linear().bias
                self.kron_weights[name] = nn.Parameter(torch.ones_like(param, **kwargs))
            else:                   # Linear().weight / Conv2d().weight
                self.kron_weights[name + '_kron0'] = nn.Parameter(torch.eye(param_size[0], **kwargs))
                self.kron_weights[name + '_kron1'] = nn.Parameter(torch.eye(param_size[1], **kwargs))
                if param_dim == 4:  # Conv2d().weight
                    self.kron_weights[name + '_kron2'] = nn.Parameter(torch.eye(param_size[2] * param_size[3], **kwargs))

    def forward(self, named_grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pointer = 0
        precond_grads = OrderedDict()

        for name, grad in named_grads.items():
            name_kron = name.replace('.', '_')
            param_dim = grad.dim()

            if param_dim == 1:  # Conv2d().bias / Linear().bias
                precond_grad = self.kron_weights[name_kron] * grad
                pointer += 1
            elif param_dim == 2:  # Linear().weight
                precond_grad = self.kron_weights[name_kron + '_kron0'] @ grad @ self.kron_weights[name_kron + '_kron1']
                pointer += 2
            elif param_dim == 4:  # Conv2d().weight
                precond_grad = torch.einsum('ijk,il->ljk',
                                            grad.flatten(start_dim=2),
                                            self.kron_weights[name_kron + '_kron0'])
                precond_grad = (self.kron_weights[name_kron + '_kron1'] @ precond_grad
                                @ self.kron_weights[name_kron + '_kron2'])
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
              first_order: bool = False) -> dict[str, nn.Parameter]:
        # TODO: better first-order implementation. Note: torch.func.grad() will not respect the outer torch.no_grad()
        #  context manager; see https://pytorch.org/docs/stable/generated/torch.func.grad.html#torch-func-grad
        batch_size = trn_inputs.size(0)
        batch_named_params = OrderedDict()
        for name, batch_param in self.meta_model['init'].named_parameters():
            batch_named_params[name] = batch_param.expand(batch_size, *batch_param.size())

        for _ in range(self.args.task_iter):
            batch_grads = self.batch_grad_fn(batch_named_params, trn_inputs, trn_targets)
            if first_order:
                batch_grads = {name: grad.detach() for name, grad in batch_grads.items()}
            batch_grads = func.vmap(self.meta_model['precond'])(batch_grads)

            for name, batch_param in batch_named_params.items():
                batch_named_params[name] = batch_param - self.args.task_lr * batch_grads[name]

        return batch_named_params
