"""Helper functions for training."""

import pyro
from pyro.infer import SVI

from model import InferPositionsPyroModel
from dataprep import DataLoader
from exceptions import NanException

import numpy as np

from typing import Tuple, List
import time
from datetime import datetime
import sys


def train_epoch(svi: SVI,
                data) -> float:
                # train_loader: DataLoader) -> float:
    """Train a single epoch.

    Args:
        svi: The pyro object used for stochastic variational inference.
        train_loader: Dataloader for training set.

    Returns:
        total_epoch_loss_train: The loss for this epoch of training, which is
            -ELBO, normalized by the number of items in the training set.

    """

    # Initialize loss accumulator and training set size.
    epoch_loss = 0.
    normalizer_train = 0.

    # # Train an epoch by going through each mini-batch.
    # for x_cell_batch in train_loader:

    # Perform gradient descent step and accumulate loss.
    epoch_loss += svi.step(data)
    normalizer_train += data.shape[0]

    # Return epoch loss.
    total_epoch_loss_train = epoch_loss / normalizer_train

    return total_epoch_loss_train


def run_training(model: InferPositionsPyroModel,
                 svi: pyro.infer.SVI,
                 data,
                 # train_loader: DataLoader,
                #  test_loader: DataLoader,
                 epochs: int,
                 learning_rate: float) -> Tuple[List[float], List[float]]:
    """Run an entire course of training, evaluating on a tests set periodically.

        Args:
            model: The model, here in order to store train and tests loss.
            args: Parsed arguments, which get saved to checkpoints.
            svi: The pyro object used for stochastic variational inference.
            train_loader: Dataloader for training set.
            test_loader: Dataloader for tests set.
            epochs: Number of epochs to run training.
            output_filename: User-specified output file, used to construct
                checkpoint filenames.
            test_freq: Test set loss is calculated every test_freq epochs of
                training.
            final_elbo_fail_fraction: Fail if final test ELBO >=
                best ELBO * (1 + this value)
            epoch_elbo_fail_fraction: Fail if current test ELBO >=
                previous ELBO * (1 + this value)
            ckpt_tarball_name: Name of saved tarball for checkpoint.
            checkpoint_freq: Checkpoint after this many minutes

        Returns:
            total_epoch_loss_train: The loss for this epoch of training, which
                is -ELBO, normalized by the number of items in the training set.

    """

    # Initialize train and tests ELBO with empty lists.
    train_elbo = []
    lr = []

    # Run training loop.  Use try to allow for keyboard interrupt.
    try:
        start_epoch = (1 if (model is None) or (len(model.loss['train']['epoch']) == 0)
                       else model.loss['train']['epoch'][-1] + 1)
        print(f'Start epoch: {start_epoch}')

        for epoch in range(start_epoch, epochs + 1):
            print(f'Training epoch {epoch}...')
            # Display duration of an epoch (use 2 to avoid initializations).
            if epoch == start_epoch + 1:
                t = time.time()

            # model.train()
            total_epoch_loss_train = train_epoch(svi, data)

            train_elbo.append(-total_epoch_loss_train)
            last_learning_rate = learning_rate
            lr.append(last_learning_rate)

            if model is not None:
                model.loss['train']['epoch'].append(epoch)
                model.loss['train']['elbo'].append(-total_epoch_loss_train)
                model.loss['learning_rate']['epoch'].append(epoch)
                model.loss['learning_rate']['value'].append(last_learning_rate)

            print(pyro.params('nuclei_locations'))
            if epoch == start_epoch + 1:
                time_per_epoch = time.time() - t
                print("[epoch %03d]  average training loss: %.4f  (%.1f seconds per epoch)"
                        % (epoch, total_epoch_loss_train, time_per_epoch))
            else:
                print("[epoch %03d]  average training loss: %.4f"
                        % (epoch, total_epoch_loss_train))


    # Exception allows program to produce output when terminated by a NaN.
    except NanException as nan:
        print(nan.message)
        print(f"Inference procedure terminated early due to a NaN value in: {nan.param}\n\n"
                f"The suggested fix is to reduce the learning rate by a factor of two.\n\n")
        sys.exit(1)

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return train_elbo