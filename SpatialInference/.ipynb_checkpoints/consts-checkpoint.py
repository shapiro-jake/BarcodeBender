"""Constant numbers used in slide-tags-map"""

# Seed for random number generators.
RANDOM_SEED = 1234

# Fraction of the data used for training (versus testing).
TRAINING_FRACTION = 0.9

# Size of minibatch (!)
DEFAULT_BATCH_SIZE = 261
SMALLEST_ALLOWED_BATCH = 100

# Prior on epsilon parameters, the RT efficiency concentration parameter [Beta(alpha, beta)]
EPSILON_CAPTURE_ALPHA_PRIOR = 5.
EPSILON_CAPTURE_BETA_PRIOR = 2.

# Prior on epsilon_perm parameters, the RT efficiency concentration parameter [Beta(alpha, beta)]
EPSILON_PERM_ALPHA_PRIOR = 5.
EPSILON_PERM_BETA_PRIOR = 2.

# Prior of rho_SB parameters, the scale factor of the beads
RHO_SB_LOC_PRIOR = 5.
RHO_SB_SCALE_PRIOR = 1.

# Prior of sigma_SB parameters, the diffusion radius of the beads
SIGMA_SB_LOC_PRIOR = 5.
SIGMA_SB_SCALE_PRIOR = 1.

# Prior of starting cell positions, center of puck
R_LOC = 3000.

# Prior of d_drop_loc and d_drop_scale
D_DROP_LOC_PRIOR = 5.86
D_DROP_SCALE_PRIOR = 0.3



# Prior of nucleus permeability
EPSILON_PERMEABILITY_PRIOR = 0.1