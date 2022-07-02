import glob
import pickle
import random
import re

import h5py
import networkx as nx
import numpy as np
import torch
import torch_geometric
from sklearn.preprocessing import StandardScaler


def natural_sort(l):
    """Sort list in Natural order.

    Parameters
    ----------
    l : {list}
        List of strings.

    Returns
    -------
    l : {ndarray}
        Sorted list of strings.
    """
    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# Create data scalers
raw_data = h5py.File('raw_data.hdf5', mode='r')

textures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

feature_data = torch.zeros((0, 5))
stiffness_data = np.empty((0, 1))
strength_data = np.empty((0, 1))

print('Loading Stiffness/Strength and Graph Data...')
for texture in textures:
    graph_files = glob.glob(f'./graphs/{texture}/*')
    graph_files = natural_sort(graph_files)

    stiffness = np.expand_dims(raw_data[f'modulus_{texture}_bc1'][()], 1)
    strength = np.expand_dims(raw_data[f'strength_{texture}_bc1'][()], 1)

    stiffness_data = np.concatenate([stiffness_data, stiffness])
    strength_data = np.concatenate([strength_data, strength])

    for i, file in enumerate(graph_files):
        G = nx.read_gpickle(file)
        data = torch_geometric.utils.from_networkx(G)
        feature_data = torch.cat([feature_data, data.x])

# Scale data
stiffness_scaler = StandardScaler()
stiffness = stiffness_scaler.fit_transform(stiffness_data)
stiffness = torch.from_numpy(stiffness)
stiffness = stiffness.to(torch.float32)

strength_scaler = StandardScaler()
strength = strength_scaler.fit_transform(strength_data)
strength = torch.from_numpy(strength)
strength = strength.to(torch.float32)

x_scaler = StandardScaler()
x_scaler.fit(feature_data.numpy())

# Save scalers
with open('./graph_data/stiffness_scaler.pickle', 'wb') as handle:
    pickle.dump(stiffness_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./graph_data/strength_scaler.pickle', 'wb') as handle:
    pickle.dump(strength_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./graph_data/x_scaler.pickle', 'wb') as handle:
    pickle.dump(x_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Scale data and split into train and test
np.random.seed(42)
random.seed(42)
rand_ids = np.random.choice(100, 100, replace=False)

leave_out = ['H', 'I', 'J', 'K', 'L']

print('Scaling and Splitting Data...')
for j, texture in enumerate(textures):
    datalist = []
    graph_files = glob.glob(f'./graphs/{texture}/*')
    graph_files = natural_sort(graph_files)

    for i, file in enumerate(graph_files):
        G = nx.read_gpickle(file)
        data = torch_geometric.utils.from_networkx(G)

        # assign scaled features
        x = x_scaler.transform(data.x.numpy())
        x = torch.from_numpy(x)
        data.x = x.to(torch.float32)
        data.stiffness = torch.unsqueeze(stiffness[i+100*j], 1)
        data.strength = torch.unsqueeze(strength[i+100*j], 1)
        datalist.append(data)

    # shuffle
    datalist_new = [datalist[i] for i in rand_ids]

    # train/test split
    if texture in leave_out:
        with open(f'./graph_data/test/{texture}.pickle', 'wb') as handle:
            pickle.dump(datalist_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f'./graph_data/train/{texture}.pickle', 'wb') as handle:
            # (90 train/val)
            pickle.dump(datalist_new[10:], handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'./graph_data/test/{texture}.pickle', 'wb') as handle:
            # (10 test)
            pickle.dump(datalist_new[:10], handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
