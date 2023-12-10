import pyro
import pyro.distributions as dist
from pyro.distributions import constraints

import torch
from torch import nn
from torch.distributions import constraints, normal, cauchy

from PoissonLog import PoissonLogParameterization

from typing import Dict, List

import consts

import numpy as np
import math


def cluster_18_model(x: torch.Tensor):
    """Cluster 18 V0 generative model for observed SBs in droplets
        
    Args:
        x: Cluster 18 data. Cell barcodes are rows, spatial barcodes are columns
    """
    
    # Hardcoded based on knowledge of cluster 18 data
    n_CBs = 203
    n_SBs = 10331

    # Calculate the ambient SB profile, which remains fixed
    chi_ambient_b = consts.CHI_AMBIENT
    
    # Load the SB locations
    SB_locations_b = torch.tensor(consts.GET_SB_LOCS)
        
    with pyro.plate("SBs", n_SBs):
        # Sample rho_SB, the scale factor of each bead
        rho_SB_loc_prior = consts.RHO_SB_LOC_PRIOR
        # rho_SB_scale_prior = consts.RHO_SB_SCALE_PRIOR
        # rho_SB_loc_b = pyro.param("rho_SB_loc_b", rho_SB_loc_prior * torch.ones(n_SBs), constraint=constraints.positive)
        # rho_SB_scale_b = pyro.param("rho_SB_scale_b", rho_SB_scale_prior * torch.ones(n_SBs), constraint=constraints.positive)
        # rho_SB_b = pyro.sample("rho_SB_b", dist.LogNormal(rho_SB_loc_b, rho_SB_scale_b))
        rho_SB_b = pyro.param("rho_SB_b", (rho_SB_loc_prior * torch.ones(n_SBs)).exp(), constraint=constraints.positive)

        sigma_SB_loc_prior = consts.SIGMA_SB_LOC_PRIOR
#         sigma_SB_scale_prior = consts.SIGMA_SB_SCALE_PRIOR
#         sigma_SB_loc_b = pyro.param("sigma_SB_loc_b", sigma_SB_loc_prior * torch.ones(n_SBs), constraint=constraints.positive)
#         sigma_SB_scale_b = pyro.param("sigma_SB_scale_b", sigma_SB_scale_prior * torch.ones(n_SBs), constraint=constraints.positive)
#         sigma_SB_b = pyro.sample("sigma_SB", dist.Normal(sigma_SB_loc_b, sigma_SB_scale_b))
        
        # Sample sigma_SB, the diffusion radius of each bead
        # sigma_SB_loc_prior = consts.SIGMA_SB_LOG_NORMAL_LOC_PRIOR
        # sigma_SB_scale_prior = consts.SIGMA_SB_LOG_NORMAL_SCALE_PRIOR
        # sigma_SB_loc_b = sigma_SB_loc_prior * torch.ones(n_SBs)
        # sigma_SB_scale_b = sigma_SB_scale_prior * torch.ones(n_SBs)
        # sigma_SB_b = pyro.sample("sigma_SB_b", dist.LogNormal(sigma_SB_loc_b, sigma_SB_scale_b)).exp()
        
        sigma_SB_b = pyro.param("sigma_SB_b", sigma_SB_loc_prior * torch.ones(n_SBs), constraint=constraints.positive)

    # Initialize priors for d_drop_loc and d_drop_scale
    d_drop_loc_prior = consts.D_DROP_LOC_PRIOR
    d_drop_scale_prior = consts.D_DROP_SCALE_PRIOR
    d_drop_loc_n = d_drop_loc_prior * torch.ones(n_CBs)
    d_drop_scale_n = d_drop_scale_prior * torch.ones(n_CBs)
    # d_drop_loc_n = pyro.param("d_drop_loc_n", d_drop_loc_prior * torch.ones(n_CBs), constraint=constraints.positive)
    # d_drop_scale_n = pyro.param("d_drop_scale_n", d_drop_scale_prior * torch.ones(n_CBs), constraint=constraints.positive)

    # Initialize priors for epsilon_capture
    epsilon_capture_alpha_prior = consts.EPSILON_CAPTURE_ALPHA_PRIOR
    epsilon_capture_beta_prior = consts.EPSILON_CAPTURE_BETA_PRIOR
    epsilon_capture_alpha_n = epsilon_capture_alpha_prior * torch.ones(n_CBs)
    epsilon_capture_beta_n = epsilon_capture_beta_prior * torch.ones(n_CBs)
    
    # Initialize priors for d_nuc_loc and d_nuc_scale
    d_nuc_loc_prior = consts.D_NUC_LOC_PRIOR
    d_nuc_scale_prior = consts.D_NUC_SCALE_PRIOR
    d_nuc_loc_n = d_nuc_loc_prior * torch.ones(n_CBs)
    d_nuc_scale_n = d_nuc_scale_prior * torch.ones(n_CBs)
    # d_nuc_loc_n = pyro.param("d_nuc_loc_n", d_nuc_loc_prior * torch.ones(n_CBs), constraint=constraints.positive)
    # d_nuc_scale_n = pyro.param("d_nuc_scale_n", d_nuc_scale_prior * torch.ones(n_CBs), constraint=constraints.positive)
    
    # One plate: nuclei; data has 203 droplets with 10331 SBs each
    with pyro.plate("data", n_CBs):
        
        # Sample nuclei locations
        radius = 2.
        nuclei_x_n = pyro.sample('nuclei_x_n', dist.Uniform(consts.R_LOC_X - radius, consts.R_LOC_X + radius))
        nuclei_y_n = pyro.sample('nuclei_y_n', dist.Uniform(consts.R_LOC_Y - radius, consts.R_LOC_Y + radius))

        
        # Calculate the absolute number of each SB at each nuclei location
        log_diff_kernel_x_nb = cauchy.Cauchy(loc=SB_locations_b[:,0], scale = sigma_SB_b).log_prob(nuclei_x_n[:, None])
        log_diff_kernel_y_nb = cauchy.Cauchy(loc=SB_locations_b[:,1], scale = sigma_SB_b).log_prob(nuclei_y_n[:, None])
        log_diff_kernel_nb = torch.log(rho_SB_b) + log_diff_kernel_x_nb + log_diff_kernel_y_nb
        log_chi_nuc_nb = log_diff_kernel_nb - torch.logsumexp(log_diff_kernel_nb, dim = -1, keepdim = True)
        
        # Sample epsilon_capture and automatically expand(n_CBs) due to plate
        epsilon_capture_n = pyro.sample("epsilon_capture_n", dist.Gamma(epsilon_capture_alpha_n, epsilon_capture_beta_n))
        
        # Sample epsilon_perm and automatically expand(n_CBs) due to plate
        d_nuc_n = pyro.sample("d_nuc_n", dist.LogNormal(d_nuc_loc_n, d_nuc_scale_n))

        # Sample d_drop and automatically expand(n_CBs) due to plate
        d_drop_n = pyro.sample("d_drop_n", dist.LogNormal(loc = d_drop_loc_n, scale = d_drop_scale_n))

        # Calculate the signal and noise rates, both of which are [261, 9497 tensors]
        log_mu_nb = torch.log(epsilon_capture_n)[:, None] + torch.log(d_nuc_n)[:, None] + log_chi_nuc_nb
        # mu = torch.zeros((n_CBs, n_SBs))
        
        log_lam_nb = torch.log(epsilon_capture_n)[:, None] + torch.log(d_drop_n)[:, None] + torch.log(chi_ambient_b)[None, :]
        
        log_rate_nb = torch.logaddexp(log_mu_nb, log_lam_nb)

        # Sample SB counts according to a Poisson distribution
        c = pyro.sample('obs_nb', PoissonLogParameterization(log_rate_nb).to_event(1), obs = x)

        return c
    
    
def cluster_18_simulation_model():
    """Cluster 18 V0 d_nuc generative model for simulating nuclei"""
    
    # Hardcoded based on knowledge of cluster 18 data
    n_CBs = 203
    n_SBs = 10331

    # Calculate the ambient SB profile, which remains fixed
    chi_ambient_b = consts.CHI_AMBIENT
    
    # Load the SB locations
    SB_locations_b = torch.tensor(consts.GET_SB_LOCS)
        
    with pyro.plate("SBs", n_SBs):
        # Sample rho_SB, the scale factor of each bead
        rho_SB_loc_prior = consts.RHO_SB_LOC_PRIOR
        rho_SB_scale_prior = consts.RHO_SB_SCALE_PRIOR
        rho_SB_loc_prior_b = rho_SB_loc_prior * torch.ones(n_SBs)
        rho_SB_scale_prior_b = rho_SB_scale_prior * torch.ones(n_SBs)
        rho_SB_b = pyro.sample("rho_SB_b", dist.LogNormal(rho_SB_loc_prior_b, rho_SB_scale_prior_b))

        # Sample sigma_SB, the diffusion radius of each bead
        # sigma_SB_loc_prior = consts.SIGMA_SB_LOC_PRIOR
        # sigma_SB_scale_prior = consts.SIGMA_SB_SCALE_PRIOR
        # sigma_SB_loc_b = sigma_SB_loc_prior * torch.ones(n_SBs)
        # sigma_SB_scale_b = sigma_SB_scale_prior * torch.ones(n_SBs)
        # sigma_SB_b = pyro.sample("sigma_SB_b", dist.Normal(sigma_SB_loc_b, sigma_SB_scale_b))
        
        sigma_SB_loc_prior = consts.SIGMA_SB_SIM
        sigma_SB_b = sigma_SB_loc_prior * torch.ones(n_SBs)

    # Initialize priors for d_drop_loc and d_drop_scale
    d_drop_loc_prior = consts.D_DROP_LOC_PRIOR
    d_drop_scale_prior = consts.D_DROP_SCALE_PRIOR
    d_drop_loc_n = d_drop_loc_prior * torch.ones(n_CBs)
    d_drop_scale_n = d_drop_scale_prior * torch.ones(n_CBs)

    # Initialize priors for epsilon_capture
    epsilon_capture_alpha_prior = consts.EPSILON_CAPTURE_ALPHA_PRIOR
    epsilon_capture_beta_prior = consts.EPSILON_CAPTURE_BETA_PRIOR
    epsilon_capture_alpha_n = epsilon_capture_alpha_prior * torch.ones(n_CBs)
    epsilon_capture_beta_n = epsilon_capture_beta_prior * torch.ones(n_CBs)
    
    # Initialize priors for d_nuc_loc and d_nuc_scale
    d_nuc_loc_prior = consts.D_NUC_LOC_PRIOR
    d_nuc_scale_prior = consts.D_NUC_SCALE_PRIOR
    d_nuc_loc_n = d_nuc_loc_prior * torch.ones(n_CBs)
    d_nuc_scale_n = d_nuc_scale_prior * torch.ones(n_CBs)
    
    # One plate: nuclei; data has 203 droplets with 10331 SBs each
    with pyro.plate("data", n_CBs):
        
        # Sample nuclei locations
        radius = 2.
        nuclei_x_n = pyro.sample('nuclei_x_n', dist.Uniform(consts.R_LOC_X - radius, consts.R_LOC_X + radius))
        nuclei_y_n = pyro.sample('nuclei_y_n', dist.Uniform(consts.R_LOC_Y - radius, consts.R_LOC_Y + radius))
        
        # Calculate the absolute number of each SB at each nuclei location
        log_diff_kernel_x_nb = normal.Normal(loc=SB_locations_b[:,0], scale = sigma_SB_b).log_prob(nuclei_x_n[:, None])
        log_diff_kernel_y_nb = normal.Normal(loc=SB_locations_b[:,1], scale = sigma_SB_b).log_prob(nuclei_y_n[:, None])
        diff_kernel_nb = rho_SB_b * (log_diff_kernel_x_nb + log_diff_kernel_y_nb).exp()
        chi_nuc_nb = diff_kernel_nb / diff_kernel_nb.sum(1)[:, None]
        
        # Sample epsilon_capture and automatically expand(n_CBs) due to plate
        epsilon_capture_n = pyro.sample("epsilon_capture_n", dist.Gamma(epsilon_capture_alpha_n, epsilon_capture_beta_n))
        
        # Sample epsilon_perm and automatically expand(n_CBs) due to plate
        d_nuc_n = pyro.sample("d_nuc_n", dist.LogNormal(d_nuc_loc_n, d_nuc_scale_n))

        # Sample d_drop and automatically expand(n_CBs) due to plate
        d_drop_n = pyro.sample("d_drop_n", dist.LogNormal(loc = d_drop_loc_n, scale = d_drop_scale_n))

        # Calculate the signal and noise rates, both of which are [203, 10331 tensors]
        mu_nb = epsilon_capture_n[:, None] * d_nuc_n[:, None] * chi_nuc_nb
        
        lam_nb = epsilon_capture_n[:, None] * d_drop_n[:, None] * chi_ambient_b[None, :]
        
        # Sample SB counts according to a Poisson distribution
        c = pyro.sample('obs_nb', dist.Poisson(rate = mu_nb + lam_nb).to_event(1))

        return c
    
    
    
    
        # max_k, k_idxs = diff_kernel_nb[0].sort(descending = True)
        # print('DIFF_KERNEL')
        # print('Max UMIs:')
        # print(max_k[:10])
        # print('\nMax rhos:')
        # print([rho_SB_b[idx] for idx in k_idxs[:10]])
        # print('\nMax radii:')
        # print([sigma_SB_b[idx] for idx in k_idxs[:10]])
        # print('\nMax locations')
        # print([SB_locations_b[idx] for idx in k_idxs[:10]])
