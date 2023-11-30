"""Constant numbers used in slide-tags-map"""

from load_h5ad import load_data
import numpy as np
import torch

# Seed for random number generators.
RANDOM_SEED = 1234

# Prior on epsilon parameters, the RT efficiency concentration parameter [Beta(alpha, beta)]
EPSILON_CAPTURE_ALPHA_PRIOR = 5.
EPSILON_CAPTURE_BETA_PRIOR = 2.

# Prior on epsilon_perm parameters, the RT efficiency concentration parameter [Beta(alpha, beta)]
EPSILON_PERM_ALPHA_PRIOR = 5.
EPSILON_PERM_BETA_PRIOR = 2.

# Prior of rho_SB parameters, the scale factor of the beads [LogNormal(loc, scale)]
RHO_SB_LOC_PRIOR = 5. # ~150
RHO_SB_SCALE_PRIOR = 1. # ~3

# Prior of sigma_SB parameters, the diffusion radius of the beads [LogNormal(loc, scale)]
SIGMA_SB_LOC_PRIOR = 0 # ~250 um
SIGMA_SB_SCALE_PRIOR = 0.5

# Prior of starting cell positions, center of puck
R_LOC = 3. # (3,000 um, 3,000 um))

# Prior of d_drop_loc and d_drop_scale [LogNormal(loc, scale)]
D_DROP_LOC_PRIOR = 5.86 # ~350, the averagenumber of SBs in an ambient droplet
D_DROP_SCALE_PRIOR = 0.3

# Calculate the ambient SB profile from pre-determined ambient droplets
def CALCULATE_AMBIENT():
    AMBIENT_DATA_H5AD_FILE = 'ambient_CB_SB_counts_top_SBs.h5ad'
    CHI_AMBIENT = np.array(load_data(AMBIENT_DATA_H5AD_FILE)['matrix'].sum(axis = 0)).squeeze()
    CHI_AMBIENT = torch.tensor(CHI_AMBIENT / CHI_AMBIENT.sum()).float()
    return CHI_AMBIENT

# Get the (x, y) coordinates, in um, of an ordered list of SBs, the same order as the SBs are listed in the h5ad files
def GET_SB_LOCS():
    ORDERED_SB_LOCATIONS = []
    with open('ordered_top_SB_coordinates.txt', 'r') as f:
        for line in f.readlines():
            SB, x_coord, y_coord = line.split(', ')
            x_coord = float(x_coord)/1000.
            y_coord = float(y_coord[:-1])/1000.
            ORDERED_SB_LOCATIONS.append([x_coord, y_coord])
    return ORDERED_SB_LOCATIONS