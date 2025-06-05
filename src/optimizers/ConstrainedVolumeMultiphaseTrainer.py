import copy
from typing import Callable

import torch.utils.data

from src.cert import Safebox
from src.optimizers.ConstrainedVolumeTrainer import ConstrainedVolumeTrainer
from src.optimizers.MultiphaseTrainer import MultiphaseTrainer
from src.utils.evaluation import evaluate_accuracy


class ConstrainedVolumeMultiphaseTrainer(MultiphaseTrainer):
    def __init__(self, trainer: ConstrainedVolumeTrainer, inflate_function: Callable[[float], float],
                 narrow_function: Callable[[float, float], float], starting_value: float, quiet: bool = False):
        super().__init__(trainer, quiet)
        self._best_acc: float | None = None
        self._inflate_function = inflate_function
        self._narrow_function = narrow_function
        self._starting_value = starting_value
        self._started = False
        self._trainer: ConstrainedVolumeTrainer = trainer
        self._volume_interval: tuple[float | None] | None = None
        self._next_volume: float | None = None
        self._best_model: torch.nn.Sequential | None = None

    def train(self, n_phases: int, val_dataset: torch.utils.data.Dataset, *trainer_args, **trainer_kwargs):
        self._print("=" * 10, f"Started Multiphase Trainer for {n_phases} phases", "=" * 10)
        if not self._started:
            self._started = True
            self._print(
                f"First time train is called. Initial training phase started with volume {self._starting_value}.")
            self._trainer.set_volume_constrain(self._starting_value)
            self._trainer.train(*trainer_args, **trainer_kwargs)
            model = Safebox.bmodelToModel(self._trainer.result())
            self._best_acc = evaluate_accuracy(val_dataset, model, num_samples=len(val_dataset))
            self._print(f"=> Initial center accuracy is {self._best_acc}")
            n_phases -= 1
            self._volume_interval = (self._starting_value, None)
            self._next_volume = self._inflate_function(self._starting_value)
            self._best_model = copy.deepcopy(model)

        for i in range(n_phases):
            self._trainer.set_volume_constrain(self._next_volume)
            self._print(f"-> Starting phase {n_phases}")
            self._print(f"-> Current Volume interval : [{round(self._volume_interval[0], 8)}, "
                        f"{round(self._volume_interval[1], 8) if self._volume_interval[1] is not None else 'INFINITY'}]")
            self._trainer.train(*trainer_args, **trainer_kwargs)
            self._print("-> Phase done")
            model = Safebox.bmodelToModel(self._trainer.result())
            acc = evaluate_accuracy(val_dataset, model, num_samples=len(val_dataset))
            self._print(f"-> Center Accuracy is {acc}.")
            if acc > self._best_acc*0.8:
                self._best_model = copy.deepcopy(model)
                self._print(f"-> Generalization improved by {acc - self._best_acc}!")
                self._best_acc = acc
                if self._volume_interval[1] is None:
                    self._volume_interval = (self._next_volume, None)
                    self._next_volume = self._inflate_function(self._next_volume)
                else:
                    self._volume_interval = (self._next_volume, self._volume_interval[1])
                    self._next_volume = self._narrow_function(*self._volume_interval)
            else:
                self._print("-> Failed to improve generalization. Reconfiguring volume interval.")
                self._volume_interval = (self._volume_interval[0], self._next_volume)
                self._next_volume = self._narrow_function(*self._volume_interval)

    def result(self) -> torch.nn.Sequential:
        return copy.deepcopy(self._best_model)
