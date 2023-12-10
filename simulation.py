import consts
from load_h5ad import load_data
from train import run_training
from plotting import plot_nuc_locs, plot_CB, plot_simulation
from model import cluster_18_model, cluster_18_simulation_model

import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide.guides import AutoDelta
from pyro.infer.autoguide.initialization import init_to_sample
import pyro.poutine as poutine
from pyro.optim import Adam


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def c18_init(site):
    if site['name'] == 'nuclei_x_n':
        print(site['name'])
        return nuclei_x_n
    if site['name'] == 'nuclei_y_n':
        print(site['name'])
        return nuclei_y_n
    return init_to_sample(site)

def infer_simulation(data, run_ID, init, gt_nuclei_x_n, gt_nuclei_y_n, epochs):
    
    # Let the guide be inferred by AutoDelta to start
    if init == 'random':
        guide = AutoDelta(cluster_18_model)
    elif init == 'optimal':
        guide = AutoDelta(cluster_18_model, c18_init)


    # Visualize model and guide
    pyro.render_model(cluster_18_model, model_args=(data,), render_distributions=True, render_params=True, filename = f'{run_ID}/{run_ID}_model_viz.png')
    pyro.render_model(guide, model_args=(data,), render_distributions=True, render_params=True, filename = f'{run_ID}/{run_ID}_guide_viz.png')

    # Loss function is ELBO
    loss_function = Trace_ELBO()

    # Set up optimizer
    learning_rate = 0.005
    betas = (0.95, 0.999)
    
    hyperparameters_file = f'{run_ID}/{run_ID}_run_settings.txt'
    print(f"Saving hyperparameters to '{hyperparameters_file}'...")
    with open(f'{run_ID}/{run_ID}_run_settings.txt', 'w') as f:
        f.write(f'{run_ID} Settings\n\n')
        f.write(f'Learning rate: {learning_rate}\n')
        f.write(f'Betas: {betas}\n')
        f.write(f'Epochs: {epochs}')
        
    optimizer_args = {'lr': learning_rate, "betas": betas}
    optimizer = pyro.optim.Adam(optimizer_args)

    # Set up SVI object
    svi = SVI(cluster_18_model, guide, optimizer, loss=loss_function)

    # Run training.
    print("Running inference...")
    
    run_training(svi=svi, data = data, epochs = epochs, run_ID = run_ID, gt_nuclei_x_n = gt_nuclei_x_n, gt_nuclei_y_n = gt_nuclei_y_n)

    print([item for item in pyro.get_param_store().items()])
    
    parameter_save_file = f'{run_ID}/{run_ID}_parameters_epoch_{epoch}.save'
    print(f"Saving parameters to '{parameter_save_file}'...")
    pyro.get_param_store().save(parameter_save_file)
    
    print("Inference procedure complete.")

if __name__ == "__main__":
    run_ID = 'test_error'
    sim = 'line'
    init = 'random'
    epochs = 200
    print_plots = False

    
    c18_CBs = load_data('slide_tags_data/cluster_18_CB_SB_counts_top_SBs.h5ad')['CBs']
    num_nuclei = len(c18_CBs)
    
    try:
        os.mkdir(run_ID)
        os.mkdir(f'{run_ID}/{run_ID}_data')
        os.mkdir(f'{run_ID}/{run_ID}_nuc_locs')
        os.mkdir(f'{run_ID}/{run_ID}_SB_scale_factors')
        os.mkdir(f'{run_ID}/{run_ID}_analysis')
    except:
        print('Directories already made...')
    

    
    if sim == False:
        # Real Data - lad Cluster 18 data and convert it into a dense tensor for training
        cluster_18_h5ad_file = 'slide_tags_data/cluster_18_CB_SB_counts_top_SBs.h5ad'
        data = load_data(cluster_18_h5ad_file)['matrix']
        data = torch.from_numpy(np.array(data.todense(), dtype=np.float32))
    else:
        if sim == 'line':
            width = 2.88
            gt_nuclei_x_n = torch.arange(consts.R_LOC_X - width / 2, consts.R_LOC_X + width / 2, width / num_nuclei)
            gt_nuclei_y_n = torch.ones(num_nuclei) * consts.R_LOC_Y
        
        elif sim == 'c18':

            ground_truth_df = pd.read_csv('slide_tags_data/gel_2_deep_cutoff_info.csv', index_col = 'CB')
            ground_truth_df = ground_truth_df[ground_truth_df['seurat_clusters'] == 18].drop('seurat_clusters', axis = 1)
            gt_nuclei_x_n, gt_nuclei_y_n = [], []
            for c18_CB in c18_CBs:
                x_coord, y_coord = ground_truth_df.loc[c18_CB] / 1000.
                gt_nuclei_x_n.append(x_coord)
                gt_nuclei_y_n.append(y_coord)

            gt_nuclei_x_n = torch.tensor(gt_nuclei_x_n)
            nuclei_y_n = torch.tensor(gt_nuclei_y_n)
        else:
            raise ValueError

        simulation_model = poutine.condition(cluster_18_simulation_model, {'nuclei_x_n': gt_nuclei_x_n, 'nuclei_y_n': gt_nuclei_y_n})
        data = simulation_model()
        
    if print_plots:
        plot_simulation(run_ID, gt_nuclei_x_n, gt_nuclei_y_n, num_nuclei, data)
    
    pyro.clear_param_store()
    infer_simulation(data, run_ID, init, gt_nuclei_x_n, gt_nuclei_y_n, epochs)
    



                                                                                