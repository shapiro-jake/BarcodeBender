import numpy as np
import pyro
import pyro.distributions as dist

import torch
from torch import nn
from torch.distributions import constraints, normal

from typing import Dict, List

import consts

import math


def model(x: torch.Tensor):
    """Cluster 18 V0 generative model for observed SBs in droplets
        
    Args:
        x: Cluster 18 data. Cell barcodes are rows, spatial barcodes are columns
    """
    
    # Hardcoded based on knowledge of cluster 18 data
    n_CBs = 261
    n_SBs = 9497

    # Calculate the ambient SB profile, which remains fixed
    chi_ambient = consts.CALCULATE_AMBIENT()
    
    # Load the SB locations
    SB_locations_b = torch.tensor(consts.GET_SB_LOCS())

    # Sample rho_SB, the scale factor of each bead
    rho_SB_loc_prior = consts.RHO_SB_LOC_PRIOR
    rho_SB_scale_prior = consts.RHO_SB_SCALE_PRIOR
    rho_SB_loc_b = rho_SB_loc_prior * torch.ones(n_SBs)
    rho_SB_scale_b = rho_SB_scale_prior * torch.ones(n_SBs)
    rho_SB_b = pyro.sample("rho_SB", dist.LogNormal(rho_SB_loc_b, rho_SB_scale_b).to_event(1)) # WHAT EXACTLY DOES TO_EVENT(1) DO HERE?
    
    # Sample sigma_SB, the diffusion radius of each bead
    sigma_SB_loc_prior = consts.SIGMA_SB_LOC_PRIOR
    sigma_SB_scale_prior = consts.SIGMA_SB_SCALE_PRIOR
    sigma_SB_loc_b = sigma_SB_loc_prior * torch.ones(n_SBs)
    sigma_SB_scale_b = sigma_SB_scale_prior * torch.ones(n_SBs)
    sigma_SB_b = pyro.sample("sigma_SB", dist.LogNormal(sigma_SB_loc_b, sigma_SB_scale_b).to_event(1))

    # Initialize nuclei locations as model parameters (?)
    nuclei_x_n = pyro.param('nuclei_x_n', consts.R_LOC * torch.ones(n_CBs))
    nuclei_y_n = pyro.param('nuclei_y_n', consts.R_LOC * torch.ones(n_CBs))

    # Initialize priors for d_drop_loc and d_drop_scale
    d_drop_loc_prior = consts.D_DROP_LOC_PRIOR
    d_drop_scale_prior = consts.D_DROP_SCALE_PRIOR
    d_drop_loc_n = d_drop_loc_prior * torch.ones(n_CBs)
    d_drop_scale_n = d_drop_scale_prior * torch.ones(n_CBs)
    
    # Calculate the absolute number of each SB at each nuclei location
    log_diff_kernel_x_nb = normal.Normal(loc=SB_locations_b[:,0], scale = sigma_SB_b).log_prob(nuclei_x_n[:, None])
    log_diff_kernel_y_nb = normal.Normal(loc=SB_locations_b[:,1], scale = sigma_SB_b).log_prob(nuclei_y_n[:, None])
    diff_kernel_nb = rho_SB_b * (log_diff_kernel_x_nb + log_diff_kernel_y_nb).exp()

    # Initialize priors for epsilon_capture
    epsilon_capture_alpha_prior = consts.EPSILON_CAPTURE_ALPHA_PRIOR
    epsilon_capture_beta_prior = consts.EPSILON_CAPTURE_BETA_PRIOR
    epsilon_capture_alpha_n = epsilon_capture_alpha_prior * torch.ones(n_CBs)
    epsilon_capture_beta_n = epsilon_capture_beta_prior * torch.ones(n_CBs)
    
    # Initialize priors for epsilon_permeability
    epsilon_perm_alpha_prior = consts.EPSILON_PERM_ALPHA_PRIOR
    epsilon_perm_beta_prior = consts.EPSILON_PERM_BETA_PRIOR
    epsilon_perm_alpha_n = epsilon_perm_alpha_prior * torch.ones(n_CBs)
    epsilon_perm_beta_n = epsilon_perm_beta_prior * torch.ones(n_CBs)
    
    # One plate: nuclei; data has 261 droplets with 9497 SBs each
    with pyro.plate("data", n_CBs):
        # Sample epsilon_capture and automatically expand(n_CBs) due to plate
        epsilon_capture_n = pyro.sample("epsilon_capture", dist.Beta(epsilon_capture_alpha_n, epsilon_capture_beta_n))

        # Sample epsilon_perm and automatically expand(n_CBs) due to plate
        epsilon_perm_n = pyro.sample("epsilon_perm", dist.Beta(epsilon_perm_alpha_n, epsilon_perm_beta_n))

        # Sample d_drop and automatically expand(n_CBs) due to plate
        d_drop_n = pyro.sample("d_drop", dist.LogNormal(loc = d_drop_loc_n, scale = d_drop_scale_n))

        # Calculate the signal and noise rates, both of which are [261, 9497 tensors]
        mu = epsilon_capture_n[:, None] * epsilon_perm_n[:, None] * diff_kernel_nb
        lam = epsilon_capture_n[:, None] * d_drop_n[:, None] * chi_ambient[None, :]

        # Sample SB counts according to a Poisson distribution
        c = pyro.sample('obs', dist.Poisson(rate = mu + lam).to_event(1), obs = x)

        return c


#     def guide(self, x: torch.Tensor):
#         """Cluster 18 V0 variational posterior
        
#         Args:
#             x: Cluster 18 data. Cell barcodes are rows, spatial barcodes are column
#         """

#         epsilon_alpha = pyro.param('epsilon_alpha', self.epsilon_alpha_prior, constraint = constraints.positive)
#         epsilon_beta = pyro.param('epsilon_beta', self.epsilon_beta_prior, constraint = constraints.positive)

#         epsilon_perm_alpha = pyro.param('epsilon_perm_alpha', self.epsilon_perm_alpha_prior, constraint = constraints.positive)
#         epsilon_perm_beta = pyro.param('epsilon_perm_beta', self.epsilon_perm_beta_prior, constraint = constraints.positive)

#         d_drop_loc = pyro.param('d_drop_loc', self.d_drop_loc_prior, constraint = constraints.positive)
#         d_drop_scale = pyro.param('d_drop_scale', self.d_drop_scale_prior, constraint = constraints.positive)

#         with pyro.plate('data', x.shape[0]):
#             epsilon = pyro.sample('epsilon',
#                                   dist.Beta(epsilon_alpha,
#                                             epsilon_beta))

#             epsilon_perm = pyro.sample('epsilon_perm', dist.Beta(epsilon_perm_alpha,
#                                                                  epsilon_perm_beta))

#             d_drop = pyro.sample('d_drop', dist.LogNormal(loc = d_drop_loc,
#                                                         scale = d_drop_scale))