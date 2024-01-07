from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all

import pyro
import pyro.distributions as dist


__all__ = ["PoissonLogParameterization"]

class TorchPoissonLogParameterization(ExponentialFamily):
    """ Creates a Poisson distribution parameterized by :attr:`log_rate`, the log of the rate parameter.
    
        Args:
            log_rate (Number, Tensor): the log of the rate parameter
    """

    @property
    def mean(self):
        return self.log_rate.exp()

    @property
    def mode(self):
        return self.log_rate.exp().floor()

    @property
    def variance(self):
        return self.log_rate.exp()

    def __init__(self, log_rate, validate_args=False):
        self.log_rate = log_rate
        # (self.log_rate,) = broadcast_all(log_rate)
        if isinstance(log_rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.log_rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_rate, value = broadcast_all(self.log_rate, value)
        return value * log_rate - log_rate.exp() - (value + 1).lgamma()

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TorchPoissonLogParameterization, _instance)
        batch_shape = torch.Size(batch_shape)
        new.log_rate = self.log_rate.expand(batch_shape)
        super(TorchPoissonLogParameterization, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.poisson(self.log_rate.exp().expand(shape))

    @property
    def _natural_params(self):
        return (torch.log(self.log_rate.exp()),)

    def _log_normalizer(self, x):
        return x

class PoissonLogParameterization(TorchPoissonLogParameterization,
                                        dist.torch_distribution.TorchDistributionMixin):
    pass