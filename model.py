import pyro
import pyro.distributions as dist
from pyro.distributions import constraints

import torch
from torch import nn
from torch.distributions import constraints, normal, cauchy, multivariate_normal

from PoissonLog import PoissonLogParameterization

from typing import Dict, List

import consts

import numpy as np
import math


def model(x: torch.Tensor):
    
    n_CBs, n_SBs = x.shape

    # Calculate the ambient SB profile, which remains fixed
    chi_ambient_b = consts.CHI_AMBIENT
    
    # Load the SB locations
    SB_locations_b2 = torch.tensor(consts.GET_SB_LOCS)
        
    with pyro.plate("SBs", n_SBs):
        # Sample rho_SB, the scale factor of each bead
        rho_SB_loc_prior = consts.RHO_SB_LOC_PRIOR
        rho_SB_scale_prior = consts.RHO_SB_SCALE_PRIOR
        rho_SB_loc_b = rho_SB_loc_prior * torch.ones(n_SBs)
        rho_SB_scale_b = rho_SB_scale_prior * torch.ones(n_SBs)
        rho_SB_b = pyro.sample("rho_SB_b", dist.LogNormal(rho_SB_loc_b, rho_SB_scale_b))
        # rho_SB_b = torch.tensor(consts.RHO_SBS)

        sigma_SB_loc_prior = consts.SIGMA_SB_LOC_PRIOR
        sigma_SB_scale_prior = consts.SIGMA_SB_SCALE_PRIOR
        sigma_SB_loc_b = sigma_SB_loc_prior * torch.ones(n_SBs)
        sigma_SB_scale_b = sigma_SB_scale_prior * torch.ones(n_SBs)
        sigma_SB_b = pyro.sample("sigma_SB_b", dist.Normal(sigma_SB_loc_b, sigma_SB_scale_b))
        # sigma_SB_b = pyro.param("sigma_SB_b", sigma_SB_loc_b, constraint = constraints.positive)
        
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
    
    # One plate: nuclei
    with pyro.plate("data", n_CBs):
        
        # Sample nuclei locations
        radius = 2.5
        nuclei_x_n = pyro.sample('nuclei_x_n', dist.Uniform(consts.R_LOC_X - radius, consts.R_LOC_X + radius))
        nuclei_y_n = pyro.sample('nuclei_y_n', dist.Uniform(consts.R_LOC_Y - radius, consts.R_LOC_Y + radius))
        
        # Calculate the absolute number of each SB at each nuclei location
        log_diff_kernel_x_nb = cauchy.Cauchy(loc=SB_locations_b2[:,0], scale = sigma_SB_b).log_prob(nuclei_x_n[:, None])
        log_diff_kernel_y_nb = cauchy.Cauchy(loc=SB_locations_b2[:,1], scale = sigma_SB_b).log_prob(nuclei_y_n[:, None])
        log_diff_kernel_nb = torch.log(rho_SB_b) + log_diff_kernel_x_nb + log_diff_kernel_y_nb
        log_chi_nuc_nb = log_diff_kernel_nb - torch.logsumexp(log_diff_kernel_nb, dim = -1, keepdim = True)
        
        # Sample epsilon_capture and automatically expand(n_CBs) due to plate
        epsilon_capture_n = pyro.sample("epsilon_capture_n", dist.Gamma(epsilon_capture_alpha_n, epsilon_capture_beta_n))
        
        # Sample epsilon_perm and automatically expand(n_CBs) due to plate
        d_nuc_n = pyro.sample("d_nuc_n", dist.Normal(d_nuc_loc_n, d_nuc_scale_n))

        # Sample d_drop and automatically expand(n_CBs) due to plate
        d_drop_n = pyro.sample("d_drop_n", dist.LogNormal(loc = d_drop_loc_n, scale = d_drop_scale_n))

        # Calculate the signal and noise rates, both of which are [261, 9497 tensors]
        log_mu_nb = torch.log(epsilon_capture_n)[:, None] + torch.log(d_nuc_n)[:, None] + log_chi_nuc_nb
        
        log_lam_nb = torch.log(epsilon_capture_n)[:, None] + torch.log(d_drop_n)[:, None] + torch.log(chi_ambient_b)[None, :]
        
        log_rate_nb = torch.logaddexp(log_mu_nb, log_lam_nb)
        
        # Sample SB counts according to a Poisson distribution
        c = pyro.sample('obs_nb', PoissonLogParameterization(log_rate_nb).to_event(1), obs = x)
        
        return c
    
    
def simulation_model():
    
    # Hardcoded based on knowledge of data
    n_CBs = 203
    n_SBs = 144678

    # Calculate the ambient SB profile, which remains fixed
    chi_ambient_b = consts.CHI_AMBIENT
    
    # Load the SB locations
    SB_locations_b2 = torch.tensor(consts.GET_SB_LOCS)
        
    with pyro.plate("SBs", n_SBs):
        # Sample rho_SB, the scale factor of each bead
        rho_SB_loc_prior = consts.RHO_SB_LOC_PRIOR
        rho_SB_scale_prior = consts.RHO_SB_SCALE_PRIOR
        rho_SB_loc_prior_b = rho_SB_loc_prior * torch.ones(n_SBs)
        rho_SB_scale_prior_b = rho_SB_scale_prior * torch.ones(n_SBs)
        rho_SB_b = pyro.sample("rho_SB_b", dist.LogNormal(rho_SB_loc_prior_b, rho_SB_scale_prior_b))
        # rho_SB_b = 100 * torch.ones(n_SBs)

        # Sample sigma_SB, the diffusion radius of each bead
        sigma_SB_loc_prior = consts.SIGMA_SB_SIM_LOC
        sigma_SB_scale_prior = consts.SIGMA_SB_SCALE_PRIOR
        sigma_SB_loc_b = sigma_SB_loc_prior * torch.ones(n_SBs)
        sigma_SB_scale_b = sigma_SB_scale_prior * torch.ones(n_SBs)
        sigma_SB_b = pyro.sample("sigma_SB_b", dist.Normal(sigma_SB_loc_b, sigma_SB_scale_b))

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
    
    with pyro.plate("data", n_CBs):
        
        # Sample nuclei locations
        radius = 2.5
        nuclei_x_n = pyro.sample('nuclei_x_n', dist.Uniform(consts.R_LOC_X - radius, consts.R_LOC_X + radius))
        nuclei_y_n = pyro.sample('nuclei_y_n', dist.Uniform(consts.R_LOC_Y - radius, consts.R_LOC_Y + radius))

        # Calculate the absolute number of each SB at each nuclei location
        log_diff_kernel_x_nb = normal.Normal(loc=SB_locations_b2[:,0], scale = sigma_SB_b).log_prob(nuclei_x_n[:, None])
        log_diff_kernel_y_nb = normal.Normal(loc=SB_locations_b2[:,1], scale = sigma_SB_b).log_prob(nuclei_y_n[:, None])
        log_diff_kernel_nb = torch.log(rho_SB_b) + log_diff_kernel_x_nb + log_diff_kernel_y_nb
        log_chi_nuc_nb = log_diff_kernel_nb - torch.logsumexp(log_diff_kernel_nb, dim = -1, keepdim = True)
        print(log_chi_nuc_nb.min(), log_chi_nuc_nb.max())
        
        # Sample epsilon_capture and automatically expand(n_CBs) due to plate
        epsilon_capture_n = pyro.sample("epsilon_capture_n", dist.Gamma(epsilon_capture_alpha_n, epsilon_capture_beta_n))
        
        # Sample epsilon_perm and automatically expand(n_CBs) due to plate
        d_nuc_n = pyro.sample("d_nuc_n", dist.LogNormal(d_nuc_loc_n, d_nuc_scale_n))

        # Sample d_drop and automatically expand(n_CBs) due to plate
        d_drop_n = pyro.sample("d_drop_n", dist.LogNormal(loc = d_drop_loc_n, scale = d_drop_scale_n))

        # Calculate the signal and noise rates, both of which are [261, 9497 tensors]
        log_mu_nb = torch.log(epsilon_capture_n)[:, None] + torch.log(d_nuc_n)[:, None] + log_chi_nuc_nb
        
        log_lam_nb = torch.log(epsilon_capture_n)[:, None] + torch.log(d_drop_n)[:, None] + torch.log(chi_ambient_b)[None, :]
        
        log_rate_nb = torch.logaddexp(log_mu_nb, log_lam_nb)
        
        # Sample SB counts according to a Poisson distribution
        c = pyro.sample('obs_nb', PoissonLogParameterization(log_rate_nb).to_event(1))
        
        return c