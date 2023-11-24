from dataset import SingleCellSBCountsDatasetV0
from model import InferPositionsPyroModel
from load_h5ad import load_data
from dataprep import prep_sparse_data_for_training
from train import run_training

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch

def run_cluster_18(cluster_18_h5ad_file, ambient_h5ad_file, ):
    ordered_SB_locations = []
    with open('../BarcodeBender/ordered_SB_coordinates.txt', 'r') as f:
        for line in f.readlines():
            SB, x_coord, y_coord = line.split(',')
            ordered_SB_locations.append((x_coord, y_coord))

    cluster_18_data = SingleCellSBCountsDatasetV0(cluster_18_h5ad_file, ambient_h5ad_file, 'cluster_18_V0', ordered_SB_locations)

    model = InferPositionsPyroModel(cluster_18_data.priors, cluster_18_data.num_CBs, cluster_18_data.num_SBs, cluster_18_data.SB_locations)

    loss_function = Trace_ELBO()

    optimizer_args = {'lr': 0.0001, "betas": (0.95, 0.999)}
    optimizer = pyro.optim.Adam(optimizer_args)

    svi = SVI(model.model, model.guide, optimizer, loss=loss_function)
    train_loader = prep_sparse_data_for_training(cluster_18_data['matrix'])

    # Run training.
    print("Running inference...")
    run_training(model=model,
                    svi=svi,
                    train_loader=train_loader,
                    epochs=10)
    
    print('Nuclei Locations:')
    print(pyro.param('nuclei_locations'))
    
    print("Inference procedure complete.")

    return model, train_loader

if __name__ == "__main__":
    cluster_18_h5ad_file = '/Users/jacobshapiro/Library/Mobile Documents/com~apple~CloudDocs/Documents/Harvard/Chen Lab/Slidetags/gel_2_deep/cluster_18_CB_SB_counts.h5ad'
    ambient_h5ad_file = '/Users/jacobshapiro/Library/Mobile Documents/com~apple~CloudDocs/Documents/Harvard/Chen Lab/Slidetags/gel_2_deep/ambient_CB_SB_counts.h5ad'
    run_cluster_18(cluster_18_h5ad_file, ambient_h5ad_file)