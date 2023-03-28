import torch
import numpy as np


class ScheduledOptimizer:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, current_step, base_lr, warmup_duration=4000, anneal_steps=[], anneal_rate=0.3):

        self._optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            #betas=[0.9, 0.98],
            #eps=10e-9,
            weight_decay=0.0,
        )
        self.n_warmup_steps = warmup_duration
        self.anneal_steps = anneal_steps
        self.anneal_rate = anneal_rate
        self.current_step = current_step
        self.init_lr = base_lr

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr