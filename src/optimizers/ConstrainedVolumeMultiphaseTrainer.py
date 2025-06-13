import copy
from typing import Callable

import torch.utils.data

from src.cert import Safebox
from src.optimizers.ConstrainedVolumeTrainer import ConstrainedVolumeTrainer
from src.optimizers.MultiphaseTrainer import MultiphaseTrainer
from src.utils.evaluation import evaluate_accuracy


class ConstrainedVolumeMultiphaseTrainer(MultiphaseTrainer):
    """
    Trainer for interval neural network models that adjusts the safe box volume in multiple phases.
    """
    def __init__(self, trainer: ConstrainedVolumeTrainer, inflate_function: Callable[[float], float],
                 narrow_function: Callable[[float, float], float], starting_value: float, quiet: bool = False,
                 min_acc_limit: float = 0.8):
        """Initializes the ConstrainedVolumeMultiphaseTrainer.

        Args:
            trainer (ConstrainedVolumeTrainer): The underlying trainer for each phase.
            inflate_function (Callable[[float], float]): Function to increase the safe box volume.
            narrow_function (Callable[[float, float], float]): Function to decrease (narrow) the safe box volume.
            starting_value (float): Initial value for the safe box volume.
            quiet (bool, optional): If True, suppresses training output. Defaults to False.
            min_acc_limit (float, optional): Minimum required accuracy for a phase to be considered successful. Defaults to 0.8.
        """
        super().__init__(trainer, quiet)
        self._inflate_function = inflate_function
        self._narrow_function = narrow_function
        self._starting_value = starting_value
        self._started = False
        self._trainer: ConstrainedVolumeTrainer = trainer
        self._volume_interval: tuple[float | None] | None = None
        self._next_volume: float | None = None
        self._best_model: torch.nn.Sequential | None = None
        self._min_acc_limit = min_acc_limit


    def train(self, n_phases: int, val_dataset: torch.utils.data.Dataset, *trainer_args, **trainer_kwargs):
        """Runs the multiphase training process, adjusting the safe box volume at each phase.

        Args:
            n_phases (int): Number of phases to run.
            val_dataset (torch.utils.data.Dataset): The validation dataset.
            *trainer_args: Additional arguments for the underlying trainer's train method.
            **trainer_kwargs: Additional keywords arguments for the underlying trainer's train method.

        Returns:
            bool: True if training succeeds and minimum accuracy is maintained, otherwise False.
        """
        self._print("=" * 10, f"Started Multiphase Trainer for {n_phases} phases", "=" * 10)
        if not self._started:
            self._started = True
            self._print(
                f"First time train is called. Initial training phase started with volume {self._starting_value}.")
            self._trainer.set_volume_constrain(self._starting_value)
            self._trainer.train(*trainer_args, **trainer_kwargs)
            model = Safebox.bmodelToModel(self._trainer.result())
            acc = evaluate_accuracy(val_dataset, model, num_samples=len(val_dataset))
            self._print(f"=> Initial center accuracy is {acc}")
            n_phases -= 1
            self._volume_interval = (self._starting_value, None)
            self._next_volume = self._inflate_function(self._starting_value)
            self._best_model = copy.deepcopy(self._trainer.result())
            if acc < self._min_acc_limit:
                self._print(f"Initial accuracy is below min accuracy threshold by {self._min_acc_limit-acc}. Training failed.")
                return False

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
            if acc > self._min_acc_limit:
                self._best_model = copy.deepcopy(self._trainer.result())
                self._print(f"-> Generalization is above minimum accuracy by {acc-self._min_acc_limit}.!")
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
        self._print("Training succeeded !")
        return True

    def result(self) -> torch.nn.Sequential:
        """Returns a copy of the best model found during training.

        Returns:
            torch.nn.Sequential: The best model.
        """
        return copy.deepcopy(self._best_model)
