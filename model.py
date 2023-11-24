import numpy as np
import pyro
import pyro.distributions as dist

import torch
from torch.distributions import constraints
from torch.distributions import normal

from scipy.stats import multivariate_normal

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
        self.n_CBs = n_CBs
        self.n_SBs = n_SBs
        self.loss = {'train': {'epoch': [], 'elbo': []},
                     'test': {'epoch': [], 'elbo': []},
                     'learning_rate': {'epoch': [], 'elbo': []}}
        
        self.d_drop_loc = torch.Tensor(dataset_obj_priors['d_drop_loc'])
        self.d_drop_scale = torch.Tensor(dataset_obj_priors['d_drop_scale'])

        self.chi_ambient = dataset_obj_priors['chi_ambient'] * torch.ones(torch.Size([]))

        self.epsilon_prior = torch.tensor(consts.EPSILON_PRIOR)

        self.epsilon_perm_prior = torch.tensor(consts.EPSILON_PERMEABILITY_PRIOR)
        self.rho_SB_prior = torch.tensor(consts.RHO_SB_PRIOR)
        self.sigma_SB_prior = torch.tensor(consts.SIGMA_SB_PRIOR)
        self.SB_locations = torch.tensor(SB_locations)
        self.nuclei_locations = torch.full((n_CBs, 2), consts.R_LOC)


    def SB_concentrations(self, nuc_locations, epsilon_perm, rho_SB, sigma_SB, SB_locations):
        """Get k_ji, the number of SBs i at the position of nucleus j"""

        concentrations = torch.Tensor((len(nuc_locations), len(SB_locations)))
        for row, nuc_location in enumerate(nuc_locations):
            for col, SB_loc in enumerate(SB_locations):
                concentrations[row][col] = epsilon_perm * rho_SB * multivariate_normal(SB_loc, sigma_SB).pdf(nuc_location)

    def model(self, x: torch.Tensor):
        """Cluster 18 V0 generative model for observed SBs in droplets
        
        Args:
            x: Cluster 18 data. Cell barcodes are rows, spatial barcodes are columns
        """

        chi_ambient = pyro.param('chi_ambient', self.chi_ambient,
                                                constraint = constraints.simplex)


        with pyro.plate('data', x.shape[0]):
            epsilon = pyro.sample('epsilon',
                                  dist.Gamma(concentration = self.epsilon_prior,
                                             rate = self.epsilon_prior)
                                        .expand_by([x.size(0)]))
            
            k = pyro.deterministic('k', self.SB_concentrations(self.nuclei_locations,
                                                               self.epsilon_perm_prior, self.rho_SB_prior,
                                                               self.SB_locations, self.sigma_SB_prior))

            d_drop = pyro.sample('d_drop', dist.LogNormal(loc = self.d_drop_loc_prior,
                                                        scale = self.d_drop_scale_prior)
                                                .expand_by([x.size(0)]))

            mu = epsilon.unsqueeze(-1) * k

            lam = epsilon.unsqueeze(-1) * d_drop.unsqueeze(-1) * chi_ambient

            c = pyro.sample('obs',
                            dist.Poisson(rate = mu + lam).to_event(1),
                            obs = x.reshape(-1, self.n_SBs))
            
            return c
            
    def guide(self, x: torch.Tensor):
        """Cluster 18 V0 variational posterior
        
        Args:
            x: Cluster 18 data. Cell barcodes are rows, spatial barcodes are column
        """

        chi_ambient = pyro.param('chi_ambient', self.chi_ambient,
                                                constraint = constraints.simplex)

        d_drop_loc = pyro.param('d_drop_loc', self.d_drop_loc, constraints = constraints.positive)
        d_drop_scale = pyro.param('d_drop_scale', self.d_drop_scale, constraints = constraints.positive)

        epsilon_perm = pyro.param('epsilon_perm', )
        rho_SB = pyro.param('rho_SB', self.rho_SB_prior, constraints = constraints.positive)
        sigma_SB = pyro.param('sigma_SB', self.sigma_SB_prior, constraints = constraints.positive)

        nuclei_locations = pyro.param('nuclei_locations', self.nuclei_locations)


        with pyro.plate('data', x.shape[0]):
            epsilon = pyro.sample('epsilon',
                                  dist.Gamma(concentration = self.epsilon_prior,
                                             rate = self.epsilon_prior)
                                        .expand_by([x.size(0)]))
            
            d_nuc = pyro.sample('d_nuc', dist.LogNormal(loc = self.d_nuc_loc_prior,
                                                        scale = self.d_nuc_scale_prior)
                                                .expand_by([x.size(0)]))

            k = pyro.deterministic('k', self.SB_concentrations(nuclei_locations,
                                                               epsilon_perm, rho_SB,
                                                               sigma_SB, self.SB_locations))

            d_drop = pyro.sample('d_drop', dist.LogNormal(loc = self.d_drop_loc_prior,
                                                        scale = self.d_drop_scale_prior)
                                                .expand_by([x.size(0)]))

            mu = epsilon.unsqueeze(-1) * d_nuc.unsequeeze(-1) * k

            lam = epsilon.unsqueeze(-1) * d_drop.unsqueeze(-1) * chi_ambient

            c = pyro.sample('obs',
                            dist.Poisson(rate = mu + lam).to_event(1),
                            obs = x.reshape(-1, self.n_SBs))