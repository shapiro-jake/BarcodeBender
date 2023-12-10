"""Helper functions for training."""

import pyro
from pyro.infer import SVI

import torch

import numpy as np

from typing import Tuple, List, Optional
import time
from datetime import datetime
import sys
from plotting import plot_nuc_locs, plot_SB_scale_factors, plot_elbo
from analysis import plot_errors



def train_epoch(svi, data) -> float:
    """Train a single epoch.

    Args:
        svi: The pyro object used for stochastic variational inference.
        data
        
    Returns:
        total_epoch_loss_train: The loss for this epoch of training, which is
            -ELBO, normalized by the number of items in the training set.
    """

    # Initialize loss accumulator and training set size.
    epoch_loss = 0.
    normalizer_train = 0.

    epoch_loss += svi.step(data)
    normalizer_train += data.size(0)


    # Return epoch loss.
    total_epoch_loss_train = epoch_loss / normalizer_train

    return total_epoch_loss_train


def run_training(svi, data, epochs, run_ID, gt_nuclei_x_n, gt_nuclei_y_n):
    """Run an entire course of training, evaluating on a tests set periodically.

        Returns:
            total_epoch_loss_train: The loss for this epoch of training, which
                is -ELBO, normalized by the number of items in the training set.

    """

    # Initialize train and tests ELBO with empty lists.
    train_elbo = []
    best_loc_error = 10**10

    # Run training loop.  Use try to allow for keyboard interrupt.
    try:

        start_epoch = 1

        for epoch in range(start_epoch, epochs + 1):

            # Display duration of an epoch (use 2 to avoid initializations).
            if epoch == start_epoch + 1:
                t = time.time()
                
                print(f'Plotting nuclei for epoch {epoch}...')
                plot_nuc_locs(epoch, run_ID)
                    
                print(f'Plotting SB scale factors for epoch {epoch}...')
                plot_SB_scale_factors(epoch, run_ID)
                
                print(f'Plotting errors for epoch {epoch}...')
                plot_errors(gt_nuclei_x_n, gt_nuclei_y_n, epoch, run_ID)
                
            total_epoch_loss_train = train_epoch(svi, data)
            train_elbo.append(-total_epoch_loss_train)

            if epoch == start_epoch + 1:
                time_per_epoch = time.time() - t
                print("[epoch %03d]  average training loss: %.4f  (%.1f seconds per epoch)"
                            % (epoch, total_epoch_loss_train, time_per_epoch))
            else:
                print("[epoch %03d]  average training loss: %.4f"
                            % (epoch, total_epoch_loss_train))
                
            if epoch % 100 == 0:
                print(f'Plotting nuclei for epoch {epoch}...')
                plot_nuc_locs(epoch, run_ID)
                    
                print(f'Plotting SB scale factors for epoch {epoch}...')
                plot_SB_scale_factors(epoch, run_ID)
                
                print(f'Plotting errors for epoch {epoch}...')
                mean_loc_error = plot_errors(gt_nuclei_x_n, gt_nuclei_y_n, epoch, run_ID)
                if mean_loc_error < best_loc_error:
                    best_parameter_save_file = f'{run_ID}/{run_ID}_parameters/{run_ID}_best_parameters_epoch_{epoch}.save'

                    print(f"Better parameters at epoch {epoch}, saving parameters to '{best_parameter_save_file}'...")
                    pyro.get_param_store().save(best_parameter_save_file)

                

    # Exception allows program to continue after ending inference prematurely.
    except KeyboardInterrupt:
        print(f"Inference procedure stopped by keyboard interrupt at epoch {epoch}... ")
        
        parameter_save_file = f'{run_ID}/{run_ID}_parameters/{run_ID}_parameters_epoch_{epoch}.save'
        print(f"Saving parameters to '{parameter_save_file}'...")
        pyro.get_param_store().save(parameter_save_file)
        
        print(f'Plotting nuclei for epoch {epoch}...')
        plot_nuc_locs(epoch, run_ID)

        print(f'Plotting SB scale factors for epoch {epoch}...')
        plot_SB_scale_factors(epoch, run_ID)
        
        print(f'Plotting errors for epoch {epoch}...')
        plot_errors(gt_nuclei_x_n, gt_nuclei_y_n, epoch, run_ID)


    print('Plotting ELBO...')
    plot_elbo(train_elbo, run_ID)
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))