"""Helper functions for preparing dataloaders, as well as
a class to implement loading data from a sparse matrix in mini-batches.

Intentionally uses global random state, and does not keep its own random number
generator, in order to facilitate checkpointing.
"""

import numpy as np
import scipy.sparse as sp

import consts

import torch
import torch.utils.data

import logging
from typing import Tuple, List, Optional, Callable


class SparseDataset(torch.utils.data.Dataset):
    """torch.utils.data.Dataset wrapping a scipy.sparse.csr.csr_matrix

    Each sample will be retrieved by indexing matrices along the leftmost
    dimension.

    Args:
        *csrs (scipy.sparse.csr.csr_matrix): sparse matrices that have the same
        size in the leftmost dimension.

    """
    # see https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html

    def __init__(self, *csrs):
        assert all(csrs[0].shape[0] == csr.shape[0] for csr in csrs)
        self.csrs = csrs

    def __getitem__(self, index) -> Tuple:
        return tuple(csr[index, ...] for csr in self.csrs)

    def __len__(self) -> int:
        return self.csrs[0].shape[0]


class DataLoader:
    """Dataloader for Cluster 18 V0 model of BarcodeBender.

    This dataloader loads all Cluster 18 CBs.
    """

    def __init__(self,
                 dataset: sp.csr_matrix,
                 batch_size: int = consts.DEFAULT_BATCH_SIZE,
                 shuffle: bool = True):
        """
        Args:
            dataset: Droplet count matrix [CB, SB]
            batch_size: Number of droplets in minibatch
            shuffle: True to shuffle data. Incompatible with sort_by.
        """

        self.dataset = dataset
        self.ind_list = np.arange(self.dataset.shape[0])
        self.sort_order = self.ind_list.copy()
        self._unsort_dict = {i: i for i in self.ind_list}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._length = None
        self._reset()

    @torch.no_grad()
    def unsort_inds(self, bcs):
        if self.sort_fn is None:
            return bcs  # just for speed
        else:
            return torch.tensor([self._unsort_dict[bc.item()] for bc in bcs], device='cpu')

    def _reset(self):
        if self.shuffle:
            np.random.shuffle(self.ind_list)  # Shuffle cell inds in place
        self.ptr = 0

    def get_state(self):
        """Internal state of the data loader, used for checkpointing"""
        return {'ind_list': self.ind_list, 'ptr': self.ptr}

    def set_state(self, ind_list: np.ndarray, ptr: int):
        self.ind_list = ind_list
        self.ptr = ptr
        assert self.ptr <= len(self.ind_list), \
            f'Problem setting dataloader state: pointer ({ptr}) is outside the ' \
            f'length of the ind_list ({len(ind_list)})'

    def reset_ptr(self):
        self.ptr = 0

    @property
    def length(self):
        if self._length is None:
            self._length = self._get_length()
        return self._length

    def _get_length(self):
        # avoid the potential for an off-by-one error by just going through it
        i = 0
        for _ in self:
            i += 1
        return i

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        # Skip last batch if the size is < smallest allowed batch
        remaining_cells = self.ind_list.size - self.ptr
        if remaining_cells < consts.SMALLEST_ALLOWED_BATCH:
            if remaining_cells > 0:
                print(f'Dropped last minibatch of {remaining_cells} cells')
            self._reset()
            raise StopIteration()

        else:

            # Move the pointer by the number of cells in this minibatch.
            next_ptr = min(self.ind_list.size, self.ptr + self.cell_batch_size)

            # Decide on CB indices.
            # cell_inds = self.ind_list[self.ptr:next_ptr]  
            csr_list = [self.dataset] #[cell_inds, :]]

            # Get a dense tensor from the sparse matrix.
            dense_tensor = sparse_collate(csr_list)

            # Increment the pointer and return the minibatch.
            self.ptr = next_ptr

            return dense_tensor


def prep_sparse_data_for_training(dataset: sp.csr_matrix,
                                #   training_fraction: float = consts.TRAINING_FRACTION,
                                  batch_size: int = consts.DEFAULT_BATCH_SIZE,
                                  shuffle: bool = True) -> torch.utils.data.DataLoader:
                                # Tuple[
                                #       torch.utils.data.DataLoader,
                                #       torch.utils.data.DataLoader]:
    """Create torch.utils.data.DataLoaders for train and tests set.

    The dataset is not loaded into memory as a dense matrix upfront.  Instead
    of using a torch.utils.data.TensorDataset, a SparseDataset is used, which
    only transforms a sparse matrix to a dense one when a minibatch is loaded.
    This is slower, but necessary for datasets which are too large to be
    loaded into memory as a dense matrix all at once.

    Args:
        dataset: Matrix of gene counts, where rows are CBs and columns are SBs.
        training_fraction: Fraction of data to use as the training set.  The
            rest becomes the test set.
        batch_size: Number of cell barcodes per mini-batch of data.
        shuffle: Passed as an argument to torch.utils.data.DataLoader.  If
            True, the data is reshuffled at every epoch.

    Returns:
        train_loader: torch.utils.data.DataLoader object for training set.
        test_loader: torch.utils.data.DataLoader object for tests set.

    Examples:
        train_loader, test_loader = prep_sparse_data_for_training(dataset,
                                        training_fraction=0.9,
                                        batch_size=128, shuffle=True)

    """

    # # Choose train and test indices from analysis dataset.
    # training_mask = np.random.rand(dataset.shape[0]) < training_fraction
    # training_indices = [idx for idx in range(dataset.shape[0])
    #                     if training_mask[idx]]
    # test_indices = [idx for idx in range(dataset.shape[0])
    #                 if not training_mask[idx]]

    # Set up training dataloader.
    # train_dataset = dataset[training_indices, ...]
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle)

    # # Set up test dataloader.
    # test_dataset = dataset[test_indices, ...]
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=batch_size,
    #                          shuffle=shuffle)

    return train_loader #, test_loader


def sparse_collate(batch: List[Tuple[sp.csr_matrix]]) -> torch.Tensor:
    """Load a minibatch of sparse data as a dense torch.Tensor in memory.

    Puts each data field into a tensor with leftmost dimension batch size.
    'batch' is a python list of items from the dataset.
    For a scipy.sparse.csr matrix, this is rows of the matrix, but in python
    list form.

    """
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html
    # default_collate()

    # Stack the list of csr matrices.
    mat = sp.vstack(batch, format='csr')
    # Output a dense torch.Tensor wrapped in a tuple.
    # This is fastest if converted in-place using torch.from_numpy().
    a = np.array(mat.todense(), dtype=np.float32)
    return torch.from_numpy(a)
