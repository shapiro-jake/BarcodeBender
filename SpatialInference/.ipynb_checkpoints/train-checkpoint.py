"""Helper functions for training."""

import pyro
from pyro.infer import SVI
import numpy as np

from exceptions import NanException
from typing import List
import time
from datetime import datetime
import sys
import random

from consts import GET_SB_LOCS
from plotting import plot_nuc_locs, plot_SB_scale_factors, plot_SB_diffusion_clouds, plot_elbo


def train_epoch(svi: SVI,
                data) -> float:
    """Train a single epoch.

    Args:
        svi: The pyro object used for stochastic variational inference.
        data: Data of cluster 18 nuclei.

    Returns:
        total_epoch_loss_train: The loss for this epoch of training, which is
            -ELBO, normalized by the number of items in the data set.

    """

    # Initialize loss accumulator and data set size.
    epoch_loss = 0.
    normalizer_train = 0.
    
    # Perform gradient descent step and accumulate loss.
    epoch_loss += svi.step(data)
    normalizer_train += data.shape[0]

    # Return epoch loss.
    total_epoch_loss_train = epoch_loss / normalizer_train

    return total_epoch_loss_train


def run_training(svi, data, epochs: int, run_ID: str) -> List[float]:
    """Run an entire course of training.

        Args:
            svi: The pyro object used for stochastic variational inference.
            data: Data of cluster 18 nuclei.
            epochs: Number of epochs to run training.
            learning_rate: The learning rate to be used

        Returns:
            train_elbo: The loss for each epoch of training

    """

    # Decide which SBs to plot diffusion clouds of
    num_SBs = len(GET_SB_LOCS())
    SB_idxs = random.choices(range(num_SBs), k = 20)

    
    # Initialize train and tests ELBO with empty lists.
    train_elbo = []

    # Run training loop.  Use try to allow for keyboard interrupt.
    try:
        start_epoch = 1
        print(f'Start epoch: {start_epoch}')

        for epoch in range(start_epoch, epochs + 1):
            print(f'Training epoch {epoch}...')
            
            # Display duration of an epoch (use 2 to avoid initializations).
            if epoch == start_epoch + 1:
                t = time.time()

            # model.train()
            total_epoch_loss_train = train_epoch(svi, data)

            train_elbo.append(-total_epoch_loss_train)

            if epoch == start_epoch + 1:
                time_per_epoch = time.time() - t
                print("[epoch %03d]  average training loss: %.4f  (%.1f seconds per epoch)"
                        % (epoch, total_epoch_loss_train, time_per_epoch))
            else:
                print("[epoch %03d]  average training loss: %.4f"
                        % (epoch, total_epoch_loss_train))
                
            if epoch % 1000 == 0:
                print(f'Plotting nuclei locations for epoch {epoch}')
                plot_nuc_locs(epoch, run_ID)
                
                print(f'Plotting SB scale factors for epoch {epoch}')
                plot_SB_scale_factors(epoch, run_ID)
                
                # Work in progress
                # print(f'SB diffusion clouds for epoch {epoch}')
                # plot_SB_diffusion_clouds(epoch, run_ID, SB_idxs)


    # Exception allows program to produce output when terminated by a NaN.
    except NanException as nan:
        print(nan.message)
        print(f"Inference procedure terminated early due to a NaN value in: {nan.param}\n\n"
                f"The suggested fix is to reduce the learning rate by a factor of two.\n\n")
        sys.exit(1)

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    plot_elbo(train_elbo, run_ID)
    
    return train_elbo