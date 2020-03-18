""" This module contains a helper class to train a model for a fixed number of
iterations, while considering early stopping based on performance from a
validation set.
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import os
import pandas as pd
import pathlib

from typing import Any

class TorchModelTrainer:
    """ This class wraps training a model with early stopping based on its
    performance on a validation set.

    Example Usage
    -------------

    ```
    model = MyModel()
    out_dir = ...
    validation_metric = 'auc'

    model_training = ModelTrainer(
        model=model,
        out_dir=out_dir,
        validation_metric=validation_metric
    )

    while model_trainer.continue_training():
        model_trainer.step()

    model_trainer.save_history()
    ```

    Members
    -------
    model : typing.Any
        An object which implements a limited version of the `ray.tune.Trainable`
        interface. In particular, the model must implement a `_train` method
        while takes no parameters are returns two dictionaries, one which
        summarizes the result of that iteration on the training set and one
        which summarizes the result of that iteration on the validation set.
        Additionally, the model must implement a `_save` method which takes
        an output directory and saves the model to that location.

    out_dir : str
        The path where the models and logs will be saved

    validation_metric : str
        The name of the validation metric. This name must appear in the
        validation dictionry returned by `model._train()`.

    optimization_mode : str
        Whether `validation_metric` should be maximized (`max`) or minimized
        (`min`)

    num_training_epochs : int
        The maximum number of training epochs

    num_patience_epochs : int
        The maximum number of epochs with no improvement of the validation
        metric before stopping training early

    name : str
        A name for this model trainer for use in logging messages
    """

    def __init__(self,
            model:Any,
            out_dir:str,
            validation_metric:str,
            optimization_mode:str='max',
            num_training_epochs:int=100,
            num_patience_epochs:int=10,
            name:str="TorchModelTrainer"):

        self.model = model
        self.out_dir = out_dir
        self.num_training_epochs = num_training_epochs
        self.validation_metric = validation_metric
        self.optimization_mode = optimization_mode
        self.num_patience_epochs = num_patience_epochs
        self.name = name

    def log(self, msg:str, level:int=logging.INFO) -> None:    
        """ Log `msg` using `level` using the module-level logger """
        msg = "[{}] {}".format(self.name, msg)
        logger.log(level, msg)
        
    def _get_epoch_performance_message(self, epoch, train_logs, val_logs) -> str:
        """ Create a message containing the keys and values in the logs """
 
        msg = 'Epoch {}:'.format(epoch)

        for k, t in train_logs.items():
            msg += ' {0}:{1:.3f} '.format(k, t)

        for k, t in val_logs.items():
            msg += ' {0}:{1:.3f} '.format(k, t)

        return msg

    def _create_output_directories(self) -> None:
        """ Create the output directories for the models and logs """
        msg = "creating output directories"
        self.log(msg)

        self._out_dir = pathlib.Path(self.out_dir)
        self._last_model_dir = self._out_dir / "last_model"
        self._best_model_dir = self._out_dir / "best_model"

        os.makedirs(self._last_model_dir, exist_ok=True)
        os.makedirs(self._best_model_dir, exist_ok=True)

    def prepare_for_training(self) -> None:
        """ Initialize the various data structures for training the model """
        self._create_output_directories()

        self._history = list()

        self._patience_count = 0
        self._current_epoch = 0
        self._best_metric = -np.inf

        # we will always maximize, so if the target metric should be minimized,
        # then the value needs to be negated to ensure maximization behaves
        # as expected
        self._sign = 1
        if self.optimization_mode == 'min':
            self._sign = -1


    def step(self) -> None:
        """ Train the model for one round
        
        Additionally, this method checks whether the performance on the
        validation metric is better than the prior best.
        """
        train_logs, val_logs = self.model._train()
        validation_value = self._sign * val_logs[self.validation_metric]

        self._patience_count += 1
        self._current_epoch += 1
        
        msg = self._get_epoch_performance_message(
            self._current_epoch, train_logs, val_logs
        )
        self.log(msg)
        
        if self._best_metric < validation_value:
            self._best_metric = validation_value
            self.model._save(self._best_model_dir)

            msg = 'Best epoch so far based on {}!'.format(self.validation_metric)
            self.log(msg)

            # since we made improvement, reset the patience counter
            self._patience_count = 0

        #scheduler.step(valid_metric2use)
            
        self._history.append(
            {**train_logs, **val_logs}
        )
        self.model._save(self._last_model_dir)


    def continue_training(self) -> bool:
        """ Check whether a stopping criterion for training has been met

        In particular, this method checks whether the maximum number of training
        rounds have been reached or if the early stopping criterion has been
        reached due to no improvement in performance on the validation set.
        """
        patience_limit = (self._patience_count > self.num_patience_epochs)
        epoch_limit = (self._current_epoch > self.num_training_epochs)

        if patience_limit:
            msg = ('No progress after {} epoch. Early stopping!'.format(
                self.num_patience_epochs)
            )
            logger.warning(msg)

        stop_training = patience_limit or epoch_limit
        continue_training = not stop_training

        return continue_training

    def get_history(self) -> pd.DataFrame:
        """ Retrieve the training and validation history as a data frame """
        # TODO: get these
        #column_names = ['BLAH']
        #column_names=list(train_logs.keys()) + list(val_logs.keys())

        df_history = pd.DataFrame(
            self._history,
            #columns=column_names,
        )

        return df_history

    def save_history(self) -> None:
        """ Write the training and validation history to disk """
        history_file = self._out_dir / "train-log.csv"
        df_history = self.get_history()
        df_history.to_csv(history_file, index=False)