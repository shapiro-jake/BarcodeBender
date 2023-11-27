import numpy as np
import pyro
import pyro.distributions as dist

import torch
from torch import nn
from torch.distributions import constraints, multivariate_normal

from typing import Dict, List

import consts

import math


class InferPositionsPyroModel(nn.Module):
    """Class that contains the model and guide used for variational inference"""

    def __init__(self,
                 dataset_obj_priors: Dict[str, float],
                 n_CBs: int,
                 n_SBs: int,
                 SB_locations: List[tuple[float]]):
        self.device = torch.device("cpu")
        self.n_CBs = n_CBs
        self.n_SBs = n_SBs
        self.loss = {'train': {'epoch': [], 'elbo': []},
                     'test': {'epoch': [], 'elbo': []},
                     'learning_rate': {'epoch': [], 'elbo': []}}
        
        self.d_drop_loc_prior = torch.Tensor([dataset_obj_priors['d_drop_loc']]).to(self.device)
        self.d_drop_scale_prior = torch.Tensor([dataset_obj_priors['d_drop_scale']]).to(self.device)

        self.chi_ambient = torch.Tensor(dataset_obj_priors['chi_ambient']).to(self.device)

        self.epsilon_alpha_prior = torch.tensor([consts.EPSILON_ALPHA_PRIOR]).to(self.device)
        self.epsilon_beta_prior = torch.tensor([consts.EPSILON_BETA_PRIOR]).to(self.device)

        self.epsilon_perm_alpha_prior = torch.tensor([consts.EPSILON_PERM_ALPHA_PRIOR]).to(self.device)
        self.epsilon_perm_beta_prior = torch.tensor([consts.EPSILON_PERM_BETA_PRIOR]).to(self.device)

        self.rho_SB_prior = torch.tensor([consts.RHO_SB_PRIOR]).to(self.device)
        self.sigma_SB_prior = torch.tensor([consts.SIGMA_SB_PRIOR]).to(self.device)
        self.SB_locations = torch.tensor(SB_locations).to(self.device)
        self.nuclei_locations_prior = torch.full((n_CBs, 2), consts.R_LOC, dtype = torch.float32).to(self.device)


    def SB_concentrations(self, nuc_locations, epsilon_perm, rho_SB, sigma_SB, SB_locations):
        """Get k_ji, the number of SBs i at the position of nucleus j"""
        concentrations = torch.empty((len(nuc_locations), len(SB_locations))).to(self.device)
        for row, nuc_location in enumerate(nuc_locations):
            for col, SB_loc in enumerate(SB_locations):
                dist = multivariate_normal.MultivariateNormal(loc=SB_loc, covariance_matrix=torch.eye(2) * sigma_SB)
                concentration = epsilon_perm[row] * rho_SB * torch.exp(dist.log_prob(nuc_location))
                concentrations[row, col] = concentration
        return concentrations

    def model(self, x: torch.Tensor):
        """Cluster 18 V0 generative model for observed SBs in droplets
        
        Args:
            x: Cluster 18 data. Cell barcodes are rows, spatial barcodes are columns
        """

        chi_ambient = pyro.param('chi_ambient', self.chi_ambient,
                                                constraint = constraints.simplex)
        
        nuclei_locations = pyro.param('nuclei_locations', self.nuclei_locations_prior)

        rho_SB = pyro.param('rho_SB', self.rho_SB_prior, constraint = constraints.positive)
        sigma_SB = pyro.param('sigma_SB', self.sigma_SB_prior, constraint = constraints.positive)
        
        with pyro.plate('data', x.shape[0]):
            epsilon = pyro.sample('epsilon',
                                  dist.Beta(self.epsilon_alpha_prior,
                                            self.epsilon_beta_prior)
                                        .expand_by([x.size(0)]))
            

            epsilon_perm = pyro.sample('epsilon_perm', dist.Beta(self.epsilon_perm_alpha_prior,
                                                                  self.epsilon_perm_beta_prior))

            k = pyro.deterministic('k', self.SB_concentrations(nuclei_locations,
                                                               epsilon_perm, rho_SB, sigma_SB,
                                                               self.SB_locations))

            d_drop = pyro.sample('d_drop', dist.LogNormal(loc = self.d_drop_loc_prior,
                                                        scale = self.d_drop_scale_prior))

            mu = epsilon.unsqueeze(-1) * k

            lam = epsilon.unsqueeze(-1) * d_drop.unsqueeze(-1) * chi_ambient

            c = pyro.sample('obs',
                            dist.Poisson(rate = mu + lam).to_event(1),
                            obs = x)
            
            return c
            
    def guide(self, x: torch.Tensor):
        """Cluster 18 V0 variational posterior
        
        Args:
            x: Cluster 18 data. Cell barcodes are rows, spatial barcodes are column
        """

        epsilon_alpha = pyro.param('epsilon_alpha', self.epsilon_alpha_prior, constraint = constraints.positive)
        epsilon_beta = pyro.param('epsilon_beta', self.epsilon_beta_prior, constraint = constraints.positive)

        epsilon_perm_alpha = pyro.param('epsilon_perm_alpha', self.epsilon_perm_alpha_prior, constraint = constraints.positive)
        epsilon_perm_beta = pyro.param('epsilon_perm_beta', self.epsilon_perm_beta_prior, constraint = constraints.positive)

        d_drop_loc = pyro.param('d_drop_loc', self.d_drop_loc_prior, constraint = constraints.positive)
        d_drop_scale = pyro.param('d_drop_scale', self.d_drop_scale_prior, constraint = constraints.positive)

        with pyro.plate('data', x.shape[0]):
            epsilon = pyro.sample('epsilon',
                                  dist.Beta(epsilon_alpha,
                                            epsilon_beta))

            epsilon_perm = pyro.sample('epsilon_perm', dist.Beta(epsilon_perm_alpha,
                                                                 epsilon_perm_beta))

            d_drop = pyro.sample('d_drop', dist.LogNormal(loc = d_drop_loc,
                                                        scale = d_drop_scale))