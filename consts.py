"""Constant numbers used in slide-tags-map"""

from load_h5ad import load_data
import numpy as np
import torch

# Seed for random number generators.
RANDOM_SEED = 1234

# Prior on epsilon parameters, the RT efficiency concentration parameter [Gamma(alpha, beta)]
EPSILON_CAPTURE_ALPHA_PRIOR = 50.
EPSILON_CAPTURE_BETA_PRIOR = 50.

# Prior on d_nuc parameters, the size of the nucleus [LogNormal(loc, scale)]
D_NUC_LOC_PRIOR = 5.2
D_NUC_SCALE_PRIOR = 0.3

# Prior of rho_SB parameters, the scale factor of the beads [LogNormal(loc, scale)]
RHO_SB_LOC_PRIOR = 1.8
RHO_SB_SCALE_PRIOR = 0.75

# Prior of sigma_SB parameters, the diffusion radius of the beads [LogNormal(loc, scale)]
SIGMA_SB_LOG_NORMAL_LOC_PRIOR = -2.6 # 75 um
SIGMA_SB_LOG_NORMAL_SCALE_PRIOR = 1.

# Prior of sigma_SB parameters, the diffusion radius of the beads [Normal(loc, scale)]
SIGMA_SB_LOC_PRIOR = 0.15 # 75 um
SIGMA_SB_SCALE_PRIOR = 0.005

# Prior of sigma_SB, the diffusion radius of the beads, for simulation
SIGMA_SB_SIM = 0.075 # 75 um

# Prior of starting cell positions, (4,250, 2,750)
R_LOC_X = 4.25
R_LOC_Y = 2.75

# Prior of d_drop_loc and d_drop_scale [LogNormal(loc, scale)]
D_DROP_LOC_PRIOR = 5.9 # the average number of SBs in an ambient droplet
D_DROP_SCALE_PRIOR = 0.3

# Calculate the ambient SB profile from pre-determined ambient droplets
def CALCULATE_AMBIENT():
    print('Calculating ambient profile...')
    AMBIENT_DATA_H5AD_FILE = 'slide_tags_data/ambient_CB_SB_counts_top_SBs.h5ad'
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
