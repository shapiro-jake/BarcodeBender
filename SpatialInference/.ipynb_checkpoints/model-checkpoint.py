import numpy as np
import pyro
import pyro.distributions as dist

import torch
from torch import nn
from torch.distributions import constraints, normal

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

        self.chi_ambient = torch.Tensor(dataset_obj_priors['chi_ambient']).to(self.device)

        self.epsilon_capture_alpha_prior = consts.EPSILON_CAPTURE_ALPHA_PRIOR
        self.epsilon_capture_beta_prior = consts.EPSILON_CAPTURE_BETA_PRIOR

        self.epsilon_perm_alpha_prior = consts.EPSILON_PERM_ALPHA_PRIOR
        self.epsilon_perm_beta_prior = consts.EPSILON_PERM_BETA_PRIOR

        self.rho_SB_loc_prior = consts.RHO_SB_LOC_PRIOR
        self.rho_SB_scale_prior = consts.RHO_SB_SCALE_PRIOR
        
        self.sigma_SB_loc_prior = consts.SIGMA_SB_LOC_PRIOR
        self.sigma_SB_scale_prior = consts.SIGMA_SB_SCALE_PRIOR
                
        self.SB_locations_b = torch.tensor(SB_locations).to(self.device)
        
        self.nuclei_x_n_prior = consts.R_LOC * torch.ones(n_CBs)
        self.nuclei_y_n_prior = consts.R_LOC * torch.ones(n_CBs)
        
        self.d_drop_loc_prior = consts.D_DROP_LOC_PRIOR
        self.d_drop_scale_prior = consts.D_DROP_SCALE_PRIOR


    def model(self, x: torch.Tensor, print_shapes = False):
        """Cluster 18 V0 generative model for observed SBs in droplets
        
        Args:
            x: Cluster 18 data. Cell barcodes are rows, spatial barcodes are columns
        """

        rho_SB_loc = self.rho_SB_loc_prior
        rho_SB_scale = self.rho_SB_scale_prior
        rho_SB_b = pyro.sample("rho_SB", dist.LogNormal(rho_SB_loc, rho_SB_scale)
                                             .expand_by(torch.Size([self.n_SBs])))
        
        sigma_SB_loc = self.sigma_SB_loc_prior
        sigma_SB_scale = self.sigma_SB_scale_prior
        sigma_SB_b = pyro.sample("sigma_SB", dist.LogNormal(sigma_SB_loc, sigma_SB_scale)
                                                 .expand_by(torch.Size([self.n_SBs])))

                                 
        nuclei_x_n = self.nuclei_x_n_prior
        nuclei_y_n = self.nuclei_y_n_prior
        log_diff_kernel_x_nb = normal.Normal(loc=self.SB_locations_b[:,0], scale = sigma_SB_b).log_prob(nuclei_x_n[:, None])
        log_diff_kernel_y_nb = normal.Normal(loc=self.SB_locations_b[:,1], scale = sigma_SB_b).log_prob(nuclei_y_n[:, None])
        diff_kernel_nb = rho_SB_b * (log_diff_kernel_x_nb + log_diff_kernel_y_nb).exp()
        
        epsilon_capture_alpha = self.epsilon_capture_alpha_prior
        epsilon_capture_beta = self.epsilon_capture_beta_prior
                                 
        epsilon_perm_alpha = self.epsilon_perm_alpha_prior
        epsilon_perm_beta = self.epsilon_perm_beta_prior
        
        d_drop_loc = self.d_drop_loc_prior
        d_drop_scale = self.d_drop_scale_prior
                                 
        with pyro.plate("data", self.n_CBs):
            epsilon_capture_n = pyro.sample("epsilon_capture", dist.Beta(epsilon_capture_alpha,
                                                                            epsilon_capture_beta)
                                                                  .expand_by([self.n_CBs]))
                                 
            epsilon_perm_n = pyro.sample("epsilon_perm", dist.Beta(epsilon_perm_alpha,
                                                                            epsilon_perm_beta)
                                                                  .expand_by([self.n_CBs]))

            d_drop_n = pyro.sample("d_drop", dist.LogNormal(loc = d_drop_loc,
                                                    scale = d_drop_scale)
                                                .expand_by([self.n_CBs]))
            
            
            mu = epsilon_capture_n[:, None] * epsilon_perm_n[:, None] * diff_kernel_nb

            lam = epsilon_capture_n[:, None] * d_drop_n[:, None] * self.chi_ambient[None, :]

            c = pyro.sample('obs',
                            dist.Poisson(rate = mu + lam).to_event(1),
                            obs = x)
            if print_shapes:
                print(f'Rho SB shape: {rho_SB_b.shape}')
                print(f'Sigma SB shape: {sigma_SB_b.shape}')
                print(f'diff_kernel_nb shape: {diff_kernel_nb.shape}')
                print(f'Epsilon capture shape: {epsilon_capture_n.shape}')
                print(f'Epsilon perm shape: {epsilon_perm_n.shape}')
                print(f'D Drop shape: {d_drop_n.shape}')
                print(f'Mu shape: {mu.shape}')
                print(f'Lam shape: {lam.shape}')
                print(f'C shape: {c.shape}')
            
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