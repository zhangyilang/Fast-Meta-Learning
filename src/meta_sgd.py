import math
from collections import OrderedDict

import torch
import torch.nn as nn

from src.meta_alg_base import MetaLearningAlgBase


class MetaSGD(MetaLearningAlgBase):
    """Meta-SGD: Learning to Learn Quickly for Few-shot Learning
    https://arxiv.org/pdf/1707.09835.pdf
    """
    def __init__(self, args):
        super().__init__(args)

    def _get_meta_model(self, **kwargs) -> dict[str, nn.Module]:
        log_lr = nn.ParameterDict()
        log_lr_init = math.log(self.args.task_lr)
        for name, param in self.base_model.meta_named_parameters():
            log_lr[name.replace('.', '_')] = nn.Parameter(torch.empty_like(param, **kwargs).fill_(log_lr_init))

        return {'init': self._get_base_model(**kwargs),
                'log_lr': log_lr}

    def adapt(self, trn_inputs: torch.Tensor, trn_targets: torch.Tensor,
              first_order: bool = False) -> dict[str, torch.nn.Parameter]:
        # TODO: better first-order implementation. Note: torch.func.grad() will not respect the outer torch.no_grad()
        #  context manager; see https://pytorch.org/docs/stable/generated/torch.func.grad.html#torch-func-grad
        batch_size = trn_inputs.size(0)
        batch_named_params = OrderedDict()
        for name, batch_param in self.meta_model['init'].named_parameters():
            batch_named_params[name] = batch_param.expand(batch_size, *batch_param.size())
        task_lr = OrderedDict({name.replace('_', '.'): log_lr.exp()
                               for name, log_lr in self.meta_model['log_lr'].items()})

        for _ in range(self.args.task_iter):
            batch_grads = self.batch_grad_fn(batch_named_params, trn_inputs, trn_targets)
            for name, batch_param in batch_named_params.items():
                batch_grad = batch_grads[name].detach() if first_order else batch_grads[name]
                batch_named_params[name] = batch_param - task_lr[name] * batch_grad

        return batch_named_params
