from model_d_nuc import cluster_18_model, cluster_18_simulation_model

import consts
import pandas as pd

from load_h5ad import load_data
from train import run_training

import torch
import pyro
import numpy as np
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide.guides import AutoDelta
from pyro.infer.autoguide.initialization import init_to_sample
import pyro.poutine as poutine
from pyro.optim import Adam
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plotting import plot_nuc_locs, plot_CB

import os


# def custom_init(site):
#     if site['name'] == 'nuclei_x_n':
#         print(site['name'])
#         return nuclei_x_n
#     if site['name'] == 'nuclei_y_n':
#         print(site['name'])
#         return nuclei_y_n
#     return init_to_sample(site)

def infer_simulation(data, run_ID):
    
    # Let the guide be inferred by AutoDelta to start
    guide = AutoDelta(cluster_18_model) #, custom_init)

    # # Visualize model and guide
    # pyro.render_model(cluster_18_model, model_args=(data,), render_distributions=True, render_params=True, filename = f'{run_ID}/{run_ID}_model_viz.png')
    # pyro.render_model(guide, model_args=(data,), render_distributions=True, render_params=True, filename = f'{run_ID}/{run_ID}_guide_viz.png')
    
    # # For debugging and visualizing shapes
    # trace = poutine.trace(cluster_18_model).get_trace(data)
    # trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    # print(trace.format_shapes())

    # Loss function is ELBO
    loss_function = Trace_ELBO()

    # Set up optimizer
    learning_rate = 0.005
    epochs = 2500
    betas = (0.95, 0.999)
    
    hyperparameters_file = f'{run_ID}/{run_ID}_run_settings.txt'
    print(f"Saving hyperparameters to '{hyperparameters_file}'...")
    with open(f'{run_ID}/{run_ID}_run_settings.txt', 'w') as f:
        f.write(f'{run_ID} Settings\n\n')
        f.write(f'Learning rate: {learning_rate}\n')
        f.write(f'Betas: {betas}\n')
        
    optimizer_args = {'lr': learning_rate, "betas": betas}
    optimizer = pyro.optim.Adam(optimizer_args)

    # Set up SVI object
    svi = SVI(cluster_18_model, guide, optimizer, loss=loss_function)

    # Run training.
    print("Running inference...")
    
    run_training(svi=svi, data = data, epochs = epochs, run_ID = run_ID)

    print([item for item in pyro.get_param_store().items()])
    
    parameter_save_file = f'{run_ID}/{run_ID}_parameters.save'
    print(f"Saving parameters to '{parameter_save_file}'...")
    pyro.get_param_store().save(parameter_save_file)
    
    print("Inference procedure complete.")

if __name__ == "__main__":
    run_ID = 'simulation_PoissonLog_CauchyDiffusion'
    try:
        os.mkdir(run_ID)
        os.mkdir(f'{run_ID}/{run_ID}_data')
    except:
        print('Directories already made...')
    

    c18_CBs = load_data('cluster_18_CB_SB_counts_top_SBs.h5ad')['CBs']
    num_nuclei = len(c18_CBs)
    
    sim = 'line'

    if sim == 'line':
        width = 2.88
        nuclei_x_n = torch.arange(consts.R_LOC_X - width / 2, consts.R_LOC_X + width / 2, width / num_nuclei)
        nuclei_y_n = torch.ones(num_nuclei) * consts.R_LOC_Y
    elif sim == 'c18':

        ground_truth_df = pd.read_csv('gel_2_deep_cutoff_info.csv', index_col = 'CB')
        ground_truth_df = ground_truth_df[ground_truth_df['seurat_clusters'] == 18]
        ground_truth_df = ground_truth_df.drop('seurat_clusters', axis = 1)
        nuclei_x_n = []
        nuclei_y_n = []
        for c18_CB in c18_CBs:
            try:
                x_coord, y_coord = ground_truth_df.loc[c18_CB]
                x_coord = x_coord/1000.
                y_coord = y_coord/1000.
                nuclei_x_n.append(x_coord)
                nuclei_y_n.append(y_coord)
            except:
                nuclei_x_n.append(4.25)
                nuclei_y_n.append(2.75)

        nuclei_x_n = torch.tensor(nuclei_x_n)
        nuclei_y_n = torch.tensor(nuclei_y_n)
    else:
        raise ValueError

    simulation_model = poutine.condition(cluster_18_simulation_model, {'nuclei_x_n': nuclei_x_n, 'nuclei_y_n': nuclei_y_n})

    data = simulation_model()

    # Real Data
    # # Load Cluster 18 data and convert it into a dense tensor for training
    # cluster_18_h5ad_file = 'cluster_18_CB_SB_counts_top_SBs.h5ad'
    # data = load_data(cluster_18_h5ad_file)['matrix']
    # data = torch.from_numpy(np.array(data.todense(), dtype=np.float32))

    print_plots = False
    
    if print_plots:
        fig, ax = plt.subplots(figsize = (6, 6))
        ax.set_title(f'{run_ID} Nuclei Locations')
        ax.set_xlabel('x_um')
        ax.set_ylabel('y_um')
        ax.set_xlim(0,6500)
        ax.set_ylim(0,6500)

        x_coords = nuclei_x_n.detach().numpy() * 1000
        y_coords = nuclei_y_n.detach().numpy() * 1000

        ax.scatter(x_coords, y_coords, s = 10, c = range(num_nuclei), cmap = 'viridis', alpha = 0.5)
        plt.savefig(f'{run_ID}/{run_ID}_data/{run_ID}_nuc_locs.png')

        SB_LOCS = consts.GET_SB_LOCS

        for i, nuc in enumerate(data):
            if i % 5 == 0:
                plot_CB(nuc, SB_LOCS, f'{run_ID}/{run_ID}_data/{run_ID}_nuc_{i}.png')

        simulated_umi_counts = np.array(data.sum(axis=1)).squeeze()
        log_simulated_umi_counts = np.log(simulated_umi_counts)
        log_simulated_umi_counts = log_simulated_umi_counts[log_simulated_umi_counts > 0]

        print(f'Max log UMI counts: {np.ceil(log_simulated_umi_counts.max())}')

        x = np.arange(
            0,
            np.ceil(log_simulated_umi_counts.max()) + 0.01,
            0.1
        )

        k = gaussian_kde(log_simulated_umi_counts)
        density = k.evaluate(x)
        log_peak_ind = np.argmax(density)
        log_peak = x[log_peak_ind]

        CB_UMI_loc = log_peak
        CB_UMI_scale = np.std(log_simulated_umi_counts)

        fig, ax = plt.subplots()
        ax.hist(log_simulated_umi_counts, bins = x)
        ax.set_ylabel('Number of Nuclei')
        ax.set_xlabel('Log(SB UMIs)')
        ax.set_title(f'Distribution of Log(SB UMIs) of {run_ID} Nuclei;\nMean: {round(CB_UMI_loc, 2)}, Std. Dev.: {round(CB_UMI_scale, 2)}')
        plt.savefig(f'{run_ID}/{run_ID}_data/{run_ID}_SB_UMI_dist.png')
    
    pyro.clear_param_store()
    infer_simulation(data, run_ID)
    



                                                                                