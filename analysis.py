import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pyro
import torch
def plot_errors(ground_truth_nuclei_x_n, ground_truth_nuclei_y_n, epoch, run_ID):
    scaled_nuclei_x_n = pyro.param('AutoDelta.nuclei_x_n') * 1000
    scaled_nuclei_y_n = pyro.param('AutoDelta.nuclei_y_n') * 1000
    scaled_ground_truth_nuclei_x_n = ground_truth_nuclei_x_n * 1000
    scaled_ground_truth_nuclei_y_n = ground_truth_nuclei_y_n * 1000
    
    delta_x_n = scaled_ground_truth_nuclei_x_n - scaled_nuclei_x_n
    delta_y_n = scaled_ground_truth_nuclei_y_n - scaled_nuclei_y_n

    errors = torch.sqrt(torch.square(delta_x_n) + torch.square(delta_y_n))
    median_error = torch.round(torch.median(errors), decimals = 2)
    mean_error = torch.round(torch.mean(errors), decimals = 2)
    
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
    fig.suptitle(f'Errors of nuclei locations at epoch {epoch} for run: {run_ID}\n Median: {median_error}, mean: {mean_error}')
    mag_ax, quiv_ax = axs
    mag_ax.set_xlabel('x_um')
    mag_ax.set_ylabel('y_um')
    mag_ax.set_xlim(0,6500)
    mag_ax.set_ylim(0,6500)
    
    quiv_ax.set_xlabel('x_um')
    quiv_ax.set_ylabel('y_um')
    quiv_ax.set_xlim(3000,6000)
    quiv_ax.set_ylim(1000,4000)
    
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

    sc = mag_ax.scatter(plotting_df['x_coords'], plotting_df['y_coords'], s = 10, c = plotting_df['errors'], cmap = 'RdBu', alpha = 1)
    fig.colorbar(sc)
    
    quiv_ax.quiver(scaled_nuclei_x_n, scaled_nuclei_y_n, delta_x_n, delta_y_n, scale = scale, color = 'cornflowerblue')
    
    plt.savefig(f'{run_ID}/{run_ID}_analysis/errors_epoch_{epoch}_{run_ID}.png')
    
    fig, hist_ax = plt.subplots(figsize = (6,6))
    hist_ax.hist(errors, bins = bins)
    hist_ax.set_ylabel('Nuclei')
    hist_ax.set_xlabel('Error')
    hist_ax.set_title(f'Distribution of Errors of {run_ID} Nuclei;\n Median: {median_error}, Mean: {mean_error}')
    plt.savefig(f'{run_ID}/{run_ID}_analysis/error_dist_epoch_{epoch}_{run_ID}.png') 
    
    plt.close()