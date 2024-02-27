import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Iterable
from src.meta_alg_base import MetaLearningAlgBase


def _log_lr(named_params: Iterable, **kwargs) -> nn.Module:
    log_lr = nn.ParameterDict()

    for name, param in named_params:
        log_lr[name.replace('.', '_')] = nn.Parameter(torch.zeros_like(param, **kwargs))

    return log_lr


class MetaSGD(MetaLearningAlgBase):
    """Meta-SGD: Learning to Learn Quickly for Few-shot Learning
    https://arxiv.org/pdf/1707.09835.pdf
    """
    def __init__(self, args):
        super().__init__(args)

    def _get_meta_model(self, **kwargs) -> dict[str, nn.Module]:
        return {'init': self._get_base_model(**kwargs),
                'log_lr': _log_lr(self.base_model.named_parameters(), **kwargs)}

    def adapt(self, trn_inputs: torch.Tensor, trn_targets: torch.Tensor,
              first_order: bool = False) -> dict[str, torch.nn.Parameter]:
        batch_size = trn_inputs.size(0)
        batch_params = OrderedDict()
        for name, param in self.meta_model['init'].named_parameters():
            batch_params[name] = param.expand(batch_size, *param.size())
        task_lr = OrderedDict({name.replace('_', '.'): log_lr.exp() * self.args.task_lr
                               for name, log_lr in self.meta_model['log_lr'].items()})

        for _ in range(self.args.task_iter):
            batch_grads = self.batch_grad_fn(batch_params, trn_inputs, trn_targets)
            for name, param in batch_params.items():
                batch_params[name] = param - task_lr[name] * batch_grads[name]

        return batch_params
