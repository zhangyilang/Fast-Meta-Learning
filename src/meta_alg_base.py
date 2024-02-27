import os
from argparse import Namespace
from abc import ABC, abstractmethod
from typing import Sequence, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.func as func
from learn2learn.data import Taskset, partition_task
from learn2learn.vision.benchmarks import get_tasksets
from learn2learn.vision.models import CNN4, ResNet12, WRN28, get_pretrained_backbone

from src.utils import Checkpointer



_BASE_MODELS = {'cnn4': CNN4,
                'resnet12': ResNet12,
                'wrn28': WRN28}


class MetaLearningAlgBase(ABC):
    """Abstract base class for meta-learning algorithms
    """
    @abstractmethod
    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.meta_trn_dataset, self.meta_val_dataset, self.meta_tst_dataset = self._get_meta_datasets()
        self.base_model = self._get_base_model(device='meta')  # dummy device
        self.meta_model = self._get_meta_model(device=self.args.device)    # real device
        self.nll = nn.CrossEntropyLoss()   # default nll for clf

        def _logit_loss_fn(params: dict[str, nn.Parameter], inputs: torch.Tensor,
                           targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            logits = func.functional_call(self.base_model, params, inputs)
            return logits, self.nll(logits, targets)

        self.batch_logit_loss_fn = func.vmap(_logit_loss_fn)
        self.batch_grad_fn = func.vmap(func.grad(lambda *argv: _logit_loss_fn(*argv)[1]))

    def _get_meta_datasets(self) -> tuple[Taskset, Taskset, Taskset]:
        return get_tasksets(self.args.dataset.lower(),
                            train_ways=self.args.num_cls,
                            train_samples=self.args.num_trn_data + self.args.num_val_data,
                            test_ways=self.args.num_cls,
                            test_samples=self.args.num_trn_data * 2,
                            root=self.args.data_dir)

    def sample_task_batch(self, meta_dataset: Taskset, num_tasks: int = 1
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_trn, batch_val = zip(*[list(partition_task(*meta_dataset.sample(),
                                                         shots=self.args.num_trn_data))
                                     for _ in range(num_tasks)])

        batch_trn_inputs, batch_trn_targets = zip(*batch_trn)
        batch_val_inputs, batch_val_targets = zip(*batch_val)

        batch_trn_inputs, batch_trn_targets = torch.stack(batch_trn_inputs), torch.stack(batch_trn_targets)
        batch_val_inputs, batch_val_targets = torch.stack(batch_val_inputs), torch.stack(batch_val_targets)

        device = self.args.device

        return (batch_trn_inputs.to(device), batch_trn_targets.to(device),
                batch_val_inputs.to(device), batch_val_targets.to(device))

    def _get_base_model(self, device: Optional[str | torch.device] = None, **kwargs) -> nn.Module:
        if self.args.base_model.lower() == 'cnn4':
            kwargs['hidden_size'] = self.args.num_filter
        model = _BASE_MODELS[self.args.base_model.lower()](output_size=self.args.num_cls, **kwargs).to(device)
        func.replace_all_batch_norm_modules_(model)     # transductive

        return model

    @abstractmethod
    def _get_meta_model(self, **kwargs) -> dict[str, nn.Module]:
        raise NotImplementedError

    def _get_meta_optimizer(self) -> tuple[optim.Optimizer, Optional[optim.lr_scheduler.LRScheduler]]:
        meta_optimizer = torch.optim.Adam([{'params': module.parameters()}
                                           for module in self.meta_model.values()],
                                          lr=self.args.meta_lr)

        return meta_optimizer, None

    @abstractmethod
    def adapt(self, trn_inputs: torch.Tensor, trn_targets: torch.Tensor,
              first_order: bool = False) -> dict[str, nn.Parameter]:
        raise NotImplementedError

    def save_meta_model(self, file_name: str) -> None:
        torch.save({name: module.state_dict() for name, module in self.meta_model.items()},
                   os.path.join(self.args.model_dir, file_name))

    def load_meta_model(self, file_name: str) -> None:
        state_dicts = torch.load(os.path.join(self.args.model_dir, file_name))
        for name, module in self.meta_model.items():
            module.load_state_dict(state_dicts[name])

    def train(self) -> None:
        print('Training starts ...')
        meta_optimizer, lr_scheduler = self._get_meta_optimizer()
        check_pointer = Checkpointer(self.save_meta_model, self.args.algorithm.lower())

        running_loss = 0.
        running_acc = 0.
        self.base_model.train()

        # training loop
        for meta_idx in range(self.args.meta_iter):
            for module in self.meta_model.values():
                module.train()
            meta_optimizer.zero_grad()

            (batch_trn_inputs, batch_trn_targets,
             batch_val_inputs, batch_val_targets) = self.sample_task_batch(self.meta_trn_dataset, self.args.batch_size)

            batch_params = self.adapt(batch_trn_inputs, batch_trn_targets, first_order=self.args.first_order)
            batch_val_logits, batch_losses = self.batch_logit_loss_fn(batch_params, batch_val_inputs, batch_val_targets)
            meta_loss = batch_losses.mean()
            meta_loss.backward()

            with torch.no_grad():
                running_loss += meta_loss.detach().item()
                running_acc += (batch_val_logits.argmax(dim=-1) == batch_val_targets).detach().float().mean().item()

            meta_optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            # meta-validation
            if (meta_idx + 1) % self.args.log_iter == 0:
                val_loss, val_acc = self.evaluate(self.meta_val_dataset, self.args.num_log_tasks)
                print(f'Meta-iter {meta_idx + 1}: '
                      f'train loss = {running_loss / self.args.log_iter:.3f}, '
                      f'train acc = {running_acc / self.args.log_iter * 100:.2f}%, '
                      f'val loss = {val_loss:.3f}, val acc = {val_acc * 100:.1f}%')

                running_loss = 0.
                running_acc = 0.

            # save
            if (meta_idx + 1) % self.args.save_iter == 0:
                val_loss, val_acc = self.evaluate(self.meta_val_dataset, self.args.num_val_tasks)
                check_pointer.update(val_acc)
                print(f'Checkpoint {check_pointer.counter}: val loss = {val_loss:.4f}, '
                      f'val acc = {val_acc * 100:.2f}%')

    def test(self) -> None:
        print('Testing starts ...')
        loss_mean, loss_95ci, acc_mean, acc_95ci = self.evaluate(self.meta_tst_dataset,
                                                                 self.args.num_tst_tasks, return_std=True)
        print(f'Test loss = {loss_mean:.4f} +/- {loss_95ci:.4f}, '
              f'test acc = {acc_mean * 100:.2f}% +/- {acc_95ci * 100:.2f}%')

    def evaluate(self, meta_dataset: Taskset, num_tasks: int, return_std=False) -> Sequence[torch.Tensor]:
        for module in self.meta_model.values():
            module.eval()   # this has no effect on the base model

        loss_list = list()
        acc_list = list()

        for eval_idx in range(num_tasks):
            (batch_trn_inputs, batch_trn_targets,
             batch_tst_inputs, batch_tst_targets) = self.sample_task_batch(meta_dataset)
            batch_params = self.adapt(batch_trn_inputs, batch_trn_targets, first_order=True)
            with torch.no_grad():
                logits, losses = self.batch_logit_loss_fn(batch_params, batch_tst_inputs, batch_tst_targets)
                loss_list.append(losses.mean().item())
                acc_list.append((logits.argmax(dim=-1) == batch_tst_targets).float().mean().item())

        if return_std:
            return (np.mean(loss_list), np.std(loss_list) * 1.96 / np.sqrt(num_tasks),
                    np.mean(acc_list), np.std(acc_list) * 1.96 / np.sqrt(num_tasks))
        else:
            return np.mean(loss_list), np.mean(acc_list)
