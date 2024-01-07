import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pyro
from pyro import condition
from pyro.poutine import trace
from model import model

import torch

def plot_errors(ground_truth_nuclei_x_n, ground_truth_nuclei_y_n, epoch, run_ID):
    
    # Plot 2D visualization of errors
    scaled_nuclei_x_n = pyro.param('AutoDelta.nuclei_x_n') * 1000
    scaled_nuclei_y_n = pyro.param('AutoDelta.nuclei_y_n') * 1000
    scaled_ground_truth_nuclei_x_n = ground_truth_nuclei_x_n * 1000
    scaled_ground_truth_nuclei_y_n = ground_truth_nuclei_y_n * 1000
    
    delta_x_n = scaled_ground_truth_nuclei_x_n - scaled_nuclei_x_n
    delta_y_n = scaled_ground_truth_nuclei_y_n - scaled_nuclei_y_n

    errors = torch.sqrt(torch.square(delta_x_n) + torch.square(delta_y_n))
    median_error = torch.round(torch.median(errors), decimals = 2)
    mean_error = torch.round(torch.mean(errors), decimals = 2)
    
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 6))
    fig.suptitle(f'Errors of nuclei locations at epoch {epoch} for run: {run_ID}\n Median: {median_error}, mean: {mean_error}')
    mag_ax, quiv_ax = axs
    mag_ax.set_xlabel('x_um')
    mag_ax.set_ylabel('y_um')
    mag_ax.set_xlim(0,6500)
    mag_ax.set_ylim(0,6500)
    
    quiv_ax.set_xlabel('x_um')
    quiv_ax.set_ylabel('y_um')
    quiv_ax.set_xlim(0,6500)
    quiv_ax.set_ylim(0,6500)
    
    scaled_nuclei_x_n = scaled_nuclei_x_n.detach().numpy()
    scaled_nuclei_y_n = scaled_nuclei_y_n.detach().numpy()
    delta_x_n = delta_x_n.detach().numpy()
    delta_y_n = delta_y_n.detach().numpy()
    scale = torch.pow(errors, -1).detach().numpy()
    
    max_error = torch.max(errors).detach().numpy()
    bins = np.arange(
        0,
        np.ceil(max_error) + 0.01,
        25
    )

    scale = 20 * max_error
    errors = errors.detach().numpy()
    
    plotting_df = pd.DataFrame(data = {'x_coords': scaled_nuclei_x_n, 'y_coords': scaled_nuclei_y_n, 'errors': errors}).sort_values('errors', ascending=True)

    sc = mag_ax.scatter(plotting_df['x_coords'], plotting_df['y_coords'], s = 5, c = plotting_df['errors'], cmap = 'RdBu', alpha = 1)
    fig.colorbar(sc)
    
    quiv_ax.quiver(scaled_nuclei_x_n, scaled_nuclei_y_n, delta_x_n, delta_y_n, scale = scale, color = 'cornflowerblue')
    
    plt.savefig(f'{run_ID}/{run_ID}_analysis/errors_epoch_{epoch}_{run_ID}.png')
    plt.close()

    # Plot histogram of errors
    fig, hist_ax = plt.subplots(figsize = (6,6))
    hist_ax.hist(errors, bins = bins)
    hist_ax.set_ylabel('Nuclei')
    hist_ax.set_xlabel('Error')
    hist_ax.set_title(f'Distribution of Errors at epoch {epoch} of {run_ID} Nuclei;\n Median: {median_error}, Mean: {mean_error}')
    plt.savefig(f'{run_ID}/{run_ID}_analysis/error_dist_epoch_{epoch}_{run_ID}.png') 
    
    plt.close()
    
    return mean_error


# def plot_log_probs(data, epoch, run_ID): # Need to change distribution from PoissonLogParameterization to Poisson for some reason...
#     print(f'Plotting log probs for epoch {epoch}...')
#     nuclei_x_n = pyro.param('AutoDelta.nuclei_x_n')
#     nuclei_y_n = pyro.param('AutoDelta.nuclei_y_n')
#     rho_SB_b = pyro.param('AutoDelta.rho_SB_b')
#     sigma_SB_b = pyro.param('AutoDelta.sigma_SB_b')
#     epsilon_capture_n = pyro.param('AutoDelta.epsilon_capture_n')
#     d_nuc_n = pyro.param('AutoDelta.d_nuc_n')
#     d_drop_n = pyro.param('AutoDelta.d_drop_n')
    
#     conditioned_model = condition(model, {
#         'nuclei_x_n': nuclei_x_n,
#         'nuclei_y_n': nuclei_y_n,
#         'rho_SB_b': rho_SB_b,
#         'sigma_SB_b': sigma_SB_b,
#         'epsilon_capture_n': epsilon_capture_n,
#         'd_nuc_n': d_nuc_n,
#         'd_drop_n': d_drop_n,
#     })
    
#     tr = trace(conditioned_model).get_trace(data)
#     tr.compute_log_prob()
#     log_prob = tr.nodes['obs_nb']['log_prob'].detach().numpy()
#     nuclei_x_n = (nuclei_x_n * 1000).detach().numpy()
#     nuclei_y_n = (nuclei_y_n * 1000).detach().numpy()
    
#     plotting_df = pd.DataFrame(data = {'x_coords': nuclei_x_n, 'y_coords': nuclei_y_n, 'log_prob': log_prob}).sort_values('log_prob', ascending=False)
    
#     fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
#     fig.suptitle(f'Log probs of nuclei at epoch {epoch} for run: {run_ID}')
#     ax.set_xlabel('x_um')
#     ax.set_ylabel('y_um')
#     ax.set_xlim(2250,6250)
#     ax.set_ylim(750,4750)
#     sc = ax.scatter(plotting_df['x_coords'], plotting_df['y_coords'], s = 5, c = plotting_df['log_prob'], cmap = 'RdBu', alpha = 1)
#     fig.colorbar(sc)
    
#     plt.savefig(f'{run_ID}/{run_ID}_analysis/log_probs_epoch_{epoch}_{run_ID}.png') 
    