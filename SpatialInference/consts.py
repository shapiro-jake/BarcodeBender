"""Constant numbers used in slide-tags-map"""

# Seed for random number generators.
RANDOM_SEED = 1234

# Fraction of the data used for training (versus testing).
TRAINING_FRACTION = 0.9

# Size of minibatch (!)
DEFAULT_BATCH_SIZE = 261
SMALLEST_ALLOWED_BATCH = 100

# Prior on epsilon parameters, the RT efficiency concentration parameter [Beta(alpha, beta)]
EPSILON_ALPHA_PRIOR = 5
EPSILON_BETA_PRIOR = 2

# Prior on epsilon_perm parameters, the RT efficiency concentration parameter [Beta(alpha, beta)]
EPSILON_PERM_ALPHA_PRIOR = 5
EPSILON_PERM_BETA_PRIOR = 2

# Prior of sigma_SB, the diffusion radius of SBs, in microns
SIGMA_SB_PRIOR = 100

# Prior of rho_SB, the scale factor of the beads
RHO_SB_PRIOR = 100

# Prior of starting cell positions, center of puck
R_LOC = 3000

# Prior of d_drop_loc and d_drop_scale
D_DROP_LOC_PRIOR = 350
D_DROP_SCALE_PRIOR = 1

# Prior of d_nuc_loc and d_nuc_scale
D_NUC_LOC_PRIOR = 72
D_NUC_SCALE_PRIOR = 0.49

# Prior of nucleus permeability
EPSILON_PERMEABILITY_PRIOR = 0.1