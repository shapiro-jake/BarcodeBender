from dataset import SingleCellSBCountsDatasetV0
from model import InferPositionsPyroModel
from load_h5ad import load_data
from dataprep import prep_sparse_data_for_training
from train import run_training

import torch
import pyro
import numpy as np
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide.guides import AutoDelta
from pyro.optim import Adam

def run_cluster_18(cluster_18_h5ad_file, ambient_h5ad_file, ):
    ordered_SB_locations = []
    with open('ordered_top_SB_coordinates.txt', 'r') as f:
        for line in f.readlines():
            SB, x_coord, y_coord = line.split(', ')
            x_coord = float(x_coord)
            y_coord = float(y_coord[:-1])
            ordered_SB_locations.append((x_coord, y_coord))

    cluster_18_data = SingleCellSBCountsDatasetV0(cluster_18_h5ad_file, ambient_h5ad_file, 'cluster_18_V0', ordered_SB_locations)
    data = cluster_18_data.data['matrix']
    data = torch.from_numpy(np.array(data.todense(), dtype=np.float32))
    
    model = InferPositionsPyroModel(cluster_18_data.priors, cluster_18_data.num_CBs, cluster_18_data.num_SBs, cluster_18_data.SB_locations)
    pyro.render_model(model.model, model_args=(data,), render_distributions=True, filename = 'model_viz.png')
    
    guide = AutoDelta(model.model)
    loss_function = Trace_ELBO()

    learning_rate = 0.0001
    optimizer_args = {'lr': learning_rate, "betas": (0.95, 0.999)}
    optimizer = pyro.optim.Adam(optimizer_args)

    svi = SVI(model.model, guide, optimizer, loss=loss_function)

    # Run training.
    print("Running inference...")
    run_training(model=model,
                    svi=svi,
                    data = data,
                    epochs=10,
                    learning_rate=learning_rate)
    
    print("Inference procedure complete.")
    print(model.loss)
    
    return model

if __name__ == "__main__":
    cluster_18_h5ad_file = 'cluster_18_CB_SB_counts_top_SBs.h5ad'
    ambient_h5ad_file = 'ambient_CB_SB_counts_top_SBs.h5ad'
    run_cluster_18(cluster_18_h5ad_file, ambient_h5ad_file)