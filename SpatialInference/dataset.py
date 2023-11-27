from typing import Dict, List, Union, Tuple
import numpy as np
import torch
from load_h5ad import load_data
import consts



class SingleCellSBCountsDatasetV0:
    """Object for storing snRNA-seq SB x CB count matrix data from Slide-Tags
    
    Currently: BarcodeBender V0
    """
    
    def __init__(self,
               mapped_input_file: str,
               ambient_input_file: str,
               model_name: str,
               SB_locations: List[Tuple[float]]
               ):

        self.mapped_input_file = mapped_input_file
        self.ambient_input_file = ambient_input_file
        self.model_name = model_name
        self.SB_locations = SB_locations

        # Load the dataset
        self.data = load_data(self.mapped_input_file)
        self.num_CBs = len(self.data['CBs'])
        self.num_SBs = len(self.data['SBs'])
        
        self.ambient_data = load_data(self.ambient_input_file)

        # Get the priors
        self.priors = self._get_priors()


    def _get_priors(self) -> Dict[str, Union[float, torch.Tensor]]:
        priors = {}
        self._estimate_chi_ambient(priors)
        self._get_empty_priors(priors)
        self._get_nuc_priors(priors)
        print(f'Priors: {priors}')
        return priors


    def _estimate_chi_ambient(self, priors):
        """Estimate chi_ambient from surely ambient droplets
        and register the value in priors dict."""
        
        chi_ambient = np.array(self.ambient_data['matrix'].sum(axis = 0)).squeeze()
        chi_ambient = torch.tensor(chi_ambient / chi_ambient.sum()).float()
        priors['chi_ambient'] = chi_ambient

    
    def _get_empty_priors(self, priors) -> Dict[str, float]:
        """Get d_drop loc and scale
        and register the values in priors dict."""

        # empty_umi_counts = np.array(self.ambient_data['matrix']
        #                   .sum(axis=1)).squeeze()

        # log_empty_umi_counts = np.log(empty_umi_counts)

        # x = np.arange(
        #     np.floor(log_empty_umi_counts.min()) - 0.01,
        #     np.ceil(log_empty_umi_counts.min()) + 0.01,
        #     0.1
        # )

        # k = guassian_kde(log_empty_umi_counts)
        # density = k.evaluate(x)
        # log_peak_ind = np.argmax(density)
        # log_peak = x[log_peak_ind]

        # d_drop_loc = np.exp(log_peak)
        # d_drop_scale = np.std(np.log(empty_umi_counts))

        # priors['d_drop_loc'] = d_drop_loc
        # priors['d_drop_scale'] = d_drop_scale

        priors['d_drop_loc'] = consts.D_DROP_LOC_PRIOR
        priors['d_drop_scale'] = consts.D_DROP_SCALE_PRIOR
    
    def _get_nuc_priors(self, priors) -> Dict[str, float]:
        """Get d_nuc loc and scale
        and register the value in priors dict.
        
        Requires that _get_empty_priors has been run
        """

        # assert 'd_nuc_loc' in self.priors.keys() and 'd_nuc_scale' in self.priors.keys()

        # nuc_umi_counts = np.array(self.data['matrix']
        #                 .sum(axis=1)).squeeze()
        
        # adjusted_nuc_umi_counts = nuc_umi_counts # - D_DROP_LOC
        # log_adjusted_nuc_umi_counts = np.log(adjusted_nuc_umi_counts)

        # x = np.arange(
        #     np.floor(log_adjusted_nuc_umi_counts.min()) - 0.01,
        #     np.ceil(log_adjusted_nuc_umi_counts.min()) + 0.01,
        #     0.1
        # )
        
        # k = guassian_kde(log_adjusted_nuc_umi_counts)
        # density = k.evaluate(x)
        # log_peak_ind = np.argmax(density)
        # log_peak = x[log_peak_ind]

        # d_nuc_loc = np.exp(log_peak)
        # d_nuc_scale = np.std(np.log(nuc_umi_counts)) ** 2 # + D_DROP_SCALE ** 2

        # priors['d_nuc_loc'] = d_nuc_loc
        # priors['d_nuc_scale'] = d_nuc_scale

        priors['epsilon_perm'] = consts.EPSILON_PERMEABILITY_PRIOR