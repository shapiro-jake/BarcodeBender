### Helper functions for plotting various things ###
import pyro

from consts import GET_SB_LOCS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import gaussian_kde

import numpy as np
import pandas as pd

def plot_simulation(run_ID, nuclei_x_n, nuclei_y_n, num_nuclei, data):
    fig, ax = plt.subplots(figsize = (6, 6))
    ax.set_title(f'{run_ID} Nuclei Locations')
    ax.set_xlabel('x_um')
    ax.set_ylabel('y_um')
    ax.set_xlim(0,6500)
    ax.set_ylim(0,6500)

    x_coords = nuclei_x_n.detach().numpy() * 1000
    y_coords = nuclei_y_n.detach().numpy() * 1000
    
    ax.scatter(x_coords, y_coords, s = 10, c = range(num_nuclei), cmap = 'viridis', alpha = 0.5)
    plt.savefig(f'{run_ID}/{run_ID}_data/{run_ID}_nuc_locs.png')

    SB_LOCS = GET_SB_LOCS

    for i, nuc in enumerate(data):
        if i % 50 == 0:
            plot_CB(nuc, SB_LOCS, f'{run_ID}/{run_ID}_data/{run_ID}_nuc_{i}.png')

    simulated_umi_counts = np.array(data.sum(axis=1)).squeeze()
    log_simulated_umi_counts = np.log(simulated_umi_counts)
    log_simulated_umi_counts = log_simulated_umi_counts[log_simulated_umi_counts > 0]

    print(f'Max log UMI counts: {np.ceil(log_simulated_umi_counts.max())}')

    x = np.arange(
        0,
        np.ceil(log_simulated_umi_counts.max()) + 0.01,
        0.1
    )

    k = gaussian_kde(log_simulated_umi_counts)
    density = k.evaluate(x)
    log_peak_ind = np.argmax(density)
    log_peak = x[log_peak_ind]

    CB_UMI_loc = log_peak
    CB_UMI_scale = np.std(log_simulated_umi_counts)

    fig, ax = plt.subplots()
    ax.hist(log_simulated_umi_counts, bins = x)
    ax.set_ylabel('Number of Nuclei')
    ax.set_xlabel('Log(SB UMIs)')
    ax.set_title(f'Distribution of Log(SB UMIs) of {run_ID} Nuclei;\nMean: {round(CB_UMI_loc, 2)}, Std. Dev.: {round(CB_UMI_scale, 2)}')
    plt.savefig(f'{run_ID}/{run_ID}_data/{run_ID}_SB_UMI_dist.png')

def plot_ground_truth(cluster_to_plot = None, savefile = None):
    nrows = 1
    ncols = 1
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (6 * ncols, 6 * nrows), constrained_layout = True)
    
    ground_truth_df = pd.read_csv('gel_2_deep_cutoff_info.csv', index_col = 'CB')
    
    if cluster_to_plot != None:
        ground_truth_df = ground_truth_df[ground_truth_df['seurat_clusters'] == cluster_to_plot]

    ground_truth_df.drop('seurat_clusters', axis = 1)
    
    ax.set_title(f'Ground Truth Based on Gel_2_Deep_Cutoff')
    ax.set_xlim(0, 6500)
    ax.set_ylim(0, 6500)
    ground_truth_df.plot.scatter(x = 'x_um', y = 'y_um', ax = ax, s = 10, alpha = 0.5, color = 'Blue')
    
    if savefile:
        plt.savefig(savefile)
    plt.close()

def plot_nuc_locs(epoch, run_ID):
    fig, ax = plt.subplots(figsize = (6, 6))
    ax.set_title(f'Nuclei locations at epoch {epoch} for run: {run_ID}')
    ax.set_xlabel('x_um')
    ax.set_ylabel('y_um')
    ax.set_xlim(0,6500)
    ax.set_ylim(0,6500)
    
    x_coords = pyro.param('AutoDelta.nuclei_x_n').detach().numpy() * 1000
    y_coords = pyro.param('AutoDelta.nuclei_y_n').detach().numpy() * 1000
    num_nuclei = len(x_coords)

    ax.scatter(x_coords, y_coords, s = 10, c = range(num_nuclei), cmap = 'viridis', alpha = 0.5)
    plt.savefig(f'{run_ID}/{run_ID}_nuc_locs/nuc_locs_epoch_{epoch}_{run_ID}.png')
    plt.close()

    
def plot_elbo(elbo, run_ID):
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('-ELBO')
    ax.set_title(f'Training ELBO for run: {run_ID}')
    ax.plot(elbo, label = '-ELBO')
    ax.legend()
    plt.savefig(f'{run_ID}/ELBO_plot_{run_ID}.png')
    plt.close()

    
def plot_SB_scale_factors(epoch, run_ID):
    fig, ax = plt.subplots()
    ax.set_title(f'Scale factor for SBs at epoch {epoch} for run: {run_ID}')
    ax.set_xlabel('x_um')
    ax.set_ylabel('y_um')
    ax.set_xlim(0,6500)
    ax.set_ylim(0,6500)
    
    rho_SBs = pyro.param('AutoDelta.rho_SB').detach().numpy()
    SB_LOCS = np.asarray(GET_SB_LOCS) * 1000
    x_SB_coords, y_SB_coords = SB_LOCS[:,0], SB_LOCS[:,1]
    plotting_df = pd.DataFrame(data = {'x_coords': x_SB_coords, 'y_coords': y_SB_coords, 'rho_SBs': rho_SBs}).sort_values('rho_SBs', ascending=True)
    sc = ax.scatter(plotting_df['x_coords'], plotting_df['y_coords'], s = 10, c = plotting_df['rho_SBs'], alpha = 0.5, cmap = 'Blues')
    
    fig.colorbar(sc)
    plt.savefig(f'{run_ID}/{run_ID}_SB_scale_factors/SB_scale_factors_epoch_{epoch}_{run_ID}.png')
    plt.close()

    
def plot_CB(counts, SB_locs, savefile = None):
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (6 * ncols, 4 * nrows), constrained_layout = True, sharex = True, sharey = True)

    # Keep track of SBs in droplet
    droplet_SB_UMIs = counts.sum()

    # Get two axes, one for raw counts, one for normalized counts
    ax_raw = axs[0]
    ax_norm = axs[1,]

    # Get SBs and initialize mapping lists
    SB_LOCS = np.asarray(SB_locs) * 1000
    x_SB_coords, y_SB_coords = SB_LOCS[:,0], SB_LOCS[:,1]

    # Normalize SB counts
    counts_normalized = counts / droplet_SB_UMIs

    # Convert to DF for sorting
    plotting_df = pd.DataFrame(data = {'x_coords': x_SB_coords, 'y_coords': y_SB_coords, 'counts': counts, 'counts_normalized': counts_normalized})
    plotting_df = plotting_df[plotting_df['counts'] > 0]

    # Plot raw UMIs
    ax_raw.set_title(f'Raw SB Counts')
    ax_raw.set_xlim(0, 6500)
    ax_raw.set_ylim(0, 6500)
    plotting_df.sort_values('counts').plot.scatter(x = 'x_coords', y = 'y_coords', c = 'counts', ax = ax_raw, s = 10, alpha = 1, cmap = 'Blues', linewidths=0.5)

    # Plot normalized UMIs
    ax_norm.set_title(f'Normalized UMIs, total SB UMIs: {droplet_SB_UMIs}')
    ax_norm.set_xlim(0, 6500)
    ax_norm.set_ylim(0, 6500)
    plotting_df.sort_values('counts_normalized').plot.scatter(x = 'x_coords', y = 'y_coords', c = 'counts_normalized', ax = ax_norm, s = 10, alpha = 1, cmap = 'Blues', linewidths=0.5)
    
    if savefile:
        plt.savefig(savefile)

    plt.close()

        
def plot_SB_diffusion_clouds(epoch, run_ID, idxs):
    fig, ax = plt.subplots()
    ax.set_title(f'Diffusion clouds for {len(idxs)} SBs at epoch {epoch} for run: {run_ID}')
    ax.set_xlabel('x_um')
    ax.set_ylabel('y_um')
    ax.set_xlim(0,6500)
    ax.set_ylim(0,6500)
    
    SB_LOCS = np.asarray(GET_SB_LOCS) * 1000
    x_SB_coords, y_SB_coords = SB_LOCS[:,0], SB_LOCS[:,1]
    
    rho_SBs = pyro.param('AutoDelta.rho_SB').detach().numpy()
    sigma_SBs = pyro.param('AutoDelta.sigma_SB').detach().numpy()
    
    for idx in idxs:
        rho_SB = rho_SBs[idx]
        sigma_SB = sigma_SBs[idx]
        x_coord, y_coord = x_SB_coords[idx], y_SB_coords[idx]
        circle = Circle((x_coord, y_coord), sigma_SB, color = rho_SB)
        ax.add_patch(circle)
    
    fig.colorbar()
    plt.savefig(f'{run_ID}/SB_diffusion_clouds_epoch_{epoch}_{run_ID}.png')
    plt.close()