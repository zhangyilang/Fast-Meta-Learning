from collections import OrderedDict
import torch
import torch.nn as nn
from src.meta_alg_base import MetaLearningAlgBase


class MAML(MetaLearningAlgBase):
    """Model-Agnostic Meta-Learning
    https://proceedings.mlr.press/v70/finn17a.html
    """
    def __init__(self, args) -> None:
        super(MAML, self).__init__(args)

    def _get_meta_model(self, **kwargs) -> dict[str, nn.Module]:
        return {'init': self._get_base_model(**kwargs)}

    def adapt(self, trn_inputs: torch.Tensor, trn_targets: torch.Tensor,
              first_order: bool = False) -> dict[str, torch.nn.Parameter]:
        batch_size = trn_inputs.size(0)
        batch_named_params = OrderedDict()
        for name, batch_param in self.meta_model['init'].named_parameters():
            batch_named_params[name] = batch_param.expand(batch_size, *batch_param.size())

        for _ in range(self.args.task_iter):
            batch_grads = self.batch_grad_fn(batch_named_params, trn_inputs, trn_targets)
            for name, batch_param in batch_named_params.items():
                batch_named_params[name] = batch_param - self.args.task_lr * batch_grads[name]

        return batch_named_params
