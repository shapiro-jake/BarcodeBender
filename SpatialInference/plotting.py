### Helper functions for plotting various things ###
import pyro
from consts import GET_SB_LOCS
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

def plot_nuc_locs(epoch, run_ID):
    fig, ax = plt.subplots()
    ax.set_title(f'Nuclei locations at epoch {epoch} for run: {run_ID}')
    ax.set_xlabel('x_um')
    ax.set_ylabel('y_um')
    ax.set_xlim(0,6500)
    ax.set_ylim(0,6500)
    
    x_coords = pyro.param('nuclei_x_n').detach().numpy() * 1000
    y_coords = pyro.param('nuclei_y_n').detach().numpy() * 1000
    num_nuclei = len(x_coords)

    ax.scatter(x_coords, y_coords, s = 10, c = range(num_nuclei), cmap = 'viridis', alpha = 0.5)
    plt.savefig(f'{run_ID}/nuc_locs_epoch_{epoch}_{run_ID}.png')
    
def plot_elbo(elbo, run_ID):
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('-ELBO')
    ax.set_title(f'Training ELBO for run: {run_ID}')
    ax.plot(elbo, label = '-ELBO')
    ax.legend()
    plt.savefig(f'{run_ID}/ELBO_plot_{run_ID}.png')
    fig.show()
    
def plot_SB_scale_factors(epoch, run_ID):
    fig, ax = plt.subplots()
    ax.set_title(f'Scale factor for SBs at epoch {epoch} for run: {run_ID}')
    ax.set_xlabel('x_um')
    ax.set_ylabel('y_um')
    ax.set_xlim(0,6500)
    ax.set_ylim(0,6500)
    
    rho_SBs = pyro.param('AutoDelta.rho_SB').detach().numpy()
    SB_LOCS = np.asarray(GET_SB_LOCS()) * 1000
    x_SB_coords, y_SB_coords = SB_LOCS[:,0], SB_LOCS[:,1]
    
    sc = ax.scatter(x_SB_coords, y_SB_coords, c = rho_SBs, cmap = 'Blues', alpha = 0.5)
    fig.colorbar(sc)
    plt.savefig(f'{run_ID}/SB_scale_factors_epoch_{epoch}_{run_ID}.png')
    
def plot_SB_diffusion_clouds(epoch, run_ID, idxs):
    fig, ax = plt.subplots()
    ax.set_title(f'Diffusion clouds for {len(idxs)} SBs at epoch {epoch} for run: {run_ID}')
    ax.set_xlabel('x_um')
    ax.set_ylabel('y_um')
    ax.set_xlim(0,6500)
    ax.set_ylim(0,6500)
    
    SB_LOCS = np.asarray(GET_SB_LOCS()) * 1000
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
    