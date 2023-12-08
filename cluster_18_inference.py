from load_h5ad import load_data
from train import run_training

import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide.guides import AutoDelta
import pyro.poutine as poutine
from pyro.optim import Adam

from model import cluster_18_model
import numpy as np
import matplotlib.pyplot as plt

from plotting import plot_nuc_locs

import os
    

def run_cluster_18(cluster_18_h5ad_file, run_ID):
    # Load Cluster 18 data and convert it into a dense tensor for training
    data = load_data(cluster_18_h5ad_file)['matrix']
    data = torch.from_numpy(np.array(data.todense(), dtype=np.float32))
    
    # Let the guide be inferred by AutoDelta to start
    guide = AutoDelta(cluster_18_model)
    print([key for key in pyro.get_param_store().keys()])

    # Visualize model and guide
    pyro.render_model(cluster_18_model, model_args=(data,), render_distributions=True, render_params=True, filename = f'{run_ID}_model_viz.png')
    pyro.render_model(guide, model_args=(data,), render_distributions=True, render_params=True, filename = f'{run_ID}_guide_viz.png')
    
    # For debugging and visualizing shapes
    trace = poutine.trace(cluster_18_model).get_trace(data)
    trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    print(trace.format_shapes())

    # Loss function is ELBO
    loss_function = Trace_ELBO()

    # Set up optimizer
    learning_rate = 0.005
    betas = (0.95, 0.999)
    optimizer_args = {'lr': learning_rate, "betas": betas}
    optimizer = pyro.optim.Adam(optimizer_args)

    # Set up SVI object
    svi = SVI(cluster_18_model, guide, optimizer, loss=loss_function)

    # Run training.
    print("Running inference...")
    
    epochs = 2500
    run_training(svi=svi, data = data, epochs = epochs, run_ID = run_ID)

    print([item for item in pyro.get_param_store().items()])
    
    parameter_save_file = f'{run_ID}/{run_ID}_parameters.save'
    print(f"Saving parameters to '{parameter_save_file}'...")
    pyro.get_param_store().save(parameter_save_file)
    
    print("Inference procedure complete.")

if __name__ == "__main__":
    run_ID = 'debug_9'
    # os.mkdir(run_ID)
    
    cluster_18_h5ad_file = 'cluster_18_CB_SB_counts_top_SBs.h5ad'
    run_cluster_18(cluster_18_h5ad_file, run_ID)