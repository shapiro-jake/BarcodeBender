import pyro
import pyro.distributions as dist
from pyro import poutine

import torch

from model import cluster_18_simulation_model
import consts

from plotting import plot_CB, plot_ground_truth
from load_h5ad import load_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# plot_ground_truth(cluster_to_plot = 18, savefile = 'cluster_18_ground_truth.png')
# plot_ground_truth(savefile = 'ground_truth.png')

SB_locs = consts.GET_SB_LOCS

unconditioned_model = poutine.uncondition(cluster_18_simulation_model)
model_trace = poutine.trace(unconditioned_model).get_trace()

print(model_trace.nodes)

# data = model_trace.nodes['obs']['value'][0]

# plot_CB(data, SB_locs, 'test_CB_plot.png')

# cluster_18_CBs = []
# with open('cluster_18_CBs.csv', 'r') as f:
#     for CB in f.readlines():
#         cluster_18_CBs.append(CB[:-1])
        
# plot_ground_truth(CBs_to_plot = cluster_18_CBs, savefile = 'cluster_18_ground_truth.png')
# plot_ground_truth(savefile = 'ground_truth.png')
        
# data = load_data('cluster_18_CB_SB_counts_top_SBs.h5ad')
# print(len(data['CBs']))
    