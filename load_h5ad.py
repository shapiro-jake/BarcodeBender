from typing import Union, Dict
import scipy.sparse as sp
import numpy as np
import anndata

def load_data(filename: str) -> Dict[str, Union[sp.csr_matrix, np.ndarray]]:
    """Load a count matrix from an h5ad AnnData file.
    The file needs to contain raw counts for all measured CBs in the
    `.X` attribute. This function returns a dictionary that includes the count matrix,
    the SBs (which correspond to columns of the count matrix),
    and the CBs (which correspond to rows of the count matrix).

    Args:
        filename: string path to .h5ad file that contains the raw CB x SB count matrix
    Returns:
        out['matrix']: scipy.sparse.csr.csr_matrix of unique SB UMI counts, with
            CBs as rows and SBs as columns
        out['CBs']: numpy array of strings which are the nucleotide
            sequences of the CBs that correspond to the rows in
            the out['matrix']
        out['SBs']: numpy array of strings which are the nucleotide sequences
            of SBs and which correspond to the columns in the out['matrix'].
    """

    adata = anndata.read_h5ad(filename)
    count_matrix = adata.X
    count_matrix = sp.csr_matrix(count_matrix)

    SBs = np.array(adata.var_names, dtype=str)
    CBs = np.array(adata.obs_names, dtype=str)

    return {'matrix': count_matrix,
            'SBs': SBs,
            'CBs': CBs}