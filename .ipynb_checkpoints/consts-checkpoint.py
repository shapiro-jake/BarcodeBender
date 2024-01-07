"""Constant numbers used in slide-tags-map"""

from load_h5ad import load_data
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# Seed for random number generators.
RANDOM_SEED = 1234

# Prior on epsilon parameters, the RT efficiency concentration parameter [Gamma(alpha, beta)]
EPSILON_CAPTURE_ALPHA_PRIOR = 50.
EPSILON_CAPTURE_BETA_PRIOR = 50.

# Prior on d_nuc parameters, the size of the nucleus [Normal/LogNormal(loc, scale)]
# For Normal
D_NUC_LOC_PRIOR = 120
D_NUC_SCALE_PRIOR = 15

# # For LogNormal
# D_NUC_LOC_PRIOR = 4.83
# D_NUC_SCALE_PRIOR = 0.5

# Prior of rho_SB parameters, the scale factor of the beads [LogNormal(loc, scale)]
RHO_SB_LOC_PRIOR = 0
RHO_SB_SCALE_PRIOR = 0.3

# Prior of sigma_SB parameters, the diffusion radius of the beads [Normal(loc, scale)]
SIGMA_SB_LOC_PRIOR = 1 # 1000 um

# Prior of sigma_SB, the diffusion radius of the beads, for simulation
SIGMA_SB_SIM_LOC = 0.025 # 25 um

# Prior of starting cell positions, (4,000, 3,000)
R_LOC_X = 3.2
R_LOC_Y = 3.25

# Prior of d_drop_loc and d_drop_scale [LogNormal(loc, scale)]
D_DROP_LOC_PRIOR = 7.099 # the average number of SBs in an ambient droplet
D_DROP_SCALE_PRIOR = 0.295

def write_priors(run_ID, priors_file):
    with open(priors_file, 'w') as f:
        f.write(f'Hyperparameters for run {run_ID}:\n\n')
        f.write(f'Epsilon capture alpha: {EPSILON_CAPTURE_ALPHA_PRIOR}\n')
        f.write(f'D_nuc_loc: {D_NUC_LOC_PRIOR}\n')
        f.write(f'D_nuc_scale: {D_NUC_SCALE_PRIOR}\n')
        f.write(f'Rho_SB_loc: {RHO_SB_LOC_PRIOR}\n')
        f.write(f'Rho_SB_scale: {RHO_SB_SCALE_PRIOR}\n')
        f.write(f'Sigma_SB_loc: {SIGMA_SB_LOC_PRIOR}\n')
        f.write(f'Sigma_SB_sim_loc: {SIGMA_SB_SIM_LOC}\n')
        f.write(f'D_drop_loc: {D_DROP_LOC_PRIOR}\n')
        f.write(f'D_drop_scale: {D_DROP_SCALE_PRIOR}\n')
        
# Calculate the ambient SB profile from pre-determined ambient droplets
def CALCULATE_AMBIENT():
    print('Calculating ambient profile...')
    AMBIENT_DATA_H5AD_FILE = 'slide_tags_data/gel_2_deep_ambient_CB_SB_counts_top_SBs.h5ad'
    CHI_AMBIENT = np.array(load_data(AMBIENT_DATA_H5AD_FILE)['matrix'].sum(axis = 0)).squeeze()
    CHI_AMBIENT = torch.tensor(CHI_AMBIENT / CHI_AMBIENT.sum()).float()
    return CHI_AMBIENT

CHI_AMBIENT = CALCULATE_AMBIENT()

# Get the (x, y) coordinates, in um, of an ordered list of SBs, the same order as the SBs are listed in the h5ad files
def GET_SB_LOCS():
    print('Getting SB locations...')
    ORDERED_SB_LOCATIONS = []
    with open('slide_tags_data/ordered_top_SB_coordinates.txt', 'r') as f:
        for line in f.readlines():
            SB, x_coord, y_coord = line.split(', ')
            x_coord = float(x_coord)/1000.
            y_coord = float(y_coord[:-1])/1000.
            ORDERED_SB_LOCATIONS.append([x_coord, y_coord])
    return ORDERED_SB_LOCATIONS

GET_SB_LOCS = GET_SB_LOCS()


# def GET_RHO_SBS(): # For using emperically-measured scale factors instead of inferring them

#     print('Getting RHO_SBs...')
#     RHO_SBs = []
#     with open('slide_tags_data/gel_2_deep_scaled_chi_ambient_rho_top_SBs.txt', 'r') as f:
#         for line in f.readlines():
#             RHO_SBs.append(np.exp(float(line)))
    
#     fig, ax = plt.subplots()
#     ax.set_title(f'Fixed scale factor for SBs')
#     ax.set_xlabel('x_um')
#     ax.set_ylabel('y_um')
#     ax.set_xlim(0,6500)
#     ax.set_ylim(0,6500)
    
#     SB_LOCS = np.asarray(GET_SB_LOCS) * 1000
#     x_SB_coords, y_SB_coords = SB_LOCS[:,0], SB_LOCS[:,1]
#     plotting_df = pd.DataFrame(data = {'x_coords': x_SB_coords, 'y_coords': y_SB_coords, 'rho_SBs': RHO_SBs}).sort_values('rho_SBs', ascending=True)
#     sc = ax.scatter(plotting_df['x_coords'], plotting_df['y_coords'], s = 5, c = np.log(plotting_df['rho_SBs']), alpha = 1, cmap = 'Blues')
    
#     fig.colorbar(sc)
#     plt.savefig(f'fixed_rho_SBs.png')
#     plt.close()
#     return RHO_SBs

# RHO_SBS = GET_RHO_SBS()