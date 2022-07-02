import pickle
import random

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import gaussian_kde
from torch_geometric.loader import DataLoader


def make_pdc(micro):
    '''Pads 3D array with opposing faces.

    Parameters
    ----------
    micro : {ndarray}
        3D array of grain IDs.

    Returns
    -------
    micro : {ndarray}
        3D array of grain IDs with periodic faces.
    '''
    micro = np.pad(micro, 1)
    micro[0, :, :] = micro[-2, :, :]
    micro[-1, :, :] = micro[1, :, :]
    micro[:, 0, :] = micro[:, -2, :]
    micro[:, -1, :] = micro[:, 1, :]
    micro[:, :, 0] = micro[:, :, -2]
    micro[:, :, -1] = micro[:, :, 1]
    return micro


def get_nbrs(micro, periodic=True):
    '''Get neighbors of grains.

    Parameters
    ----------
    micro : {ndarray}
        3D array of grain IDs.
        *DONT USE grain ID == 0*

    Returns
    -------
    nbr_dict : {dictionary} {grain_ID : [nbrs, shared_area]}
        Dictionary giving neighbors and shared area of each neighbor for every grain.
    '''
    if periodic:
        micro = make_pdc(micro)
    else:
        micro = np.pad(micro, 1)

    dim = micro.shape

    # structure element used to get voxel face neighbors
    s = np.zeros((3, 3, 3))
    s[1, 1, :] = 1
    s[1, :, 1] = 1
    s[:, 1, 1] = 1
    s[1, 1, 1] = 0

    nbr_dict = {}
    for feat in np.unique(micro):
        if feat == 0:
            continue
        nbr_list = []
        for x in range(1, dim[0]-1):
            for y in range(1, dim[1]-1):
                for z in range(1, dim[2]-1):
                    if micro[x, y, z] == feat:
                        nbrs = s*micro[x-1:x+2, y-1:y+2, z-1:z+2]
                        nbrs = nbrs[~np.isin(nbrs, [0, feat])]
                        for nbr in nbrs:
                            nbr_list.append(nbr)

        nbrs, counts = np.unique(np.asarray(nbr_list), return_counts=True)
        nbr_dict[feat] = [nbrs.astype(int), counts]
    return nbr_dict


def assign_grains(micro):
    '''Get grain IDs from microstructure.

    Parameters
    ----------
    micro : {ndarray} of shape (n, n, n, ori_dim)
        4D array of orientations. quaternions (ori_dim=4)

    Returns
    -------
    grain_ids : {dictionary} of shape (n, n, n)
        3D array of grain IDs.

    grain_oris : {ndarray}
        Array of unique grain orientations.
    '''
    dims = micro.shape[1:]
    grain_ids = np.zeros(dims, dtype=np.int32)
    ori_list = []
    grn_cnt = 1
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                ori = micro[:, i, j, k].tolist()
                if ori not in ori_list:
                    grain_ids[i, j, k] = grn_cnt
                    ori_list.append(ori)
                    grn_cnt += 1
                else:
                    grain_ids[i, j, k] = ori_list.index(ori)+1
    grain_oris = np.asarray(ori_list)
    return grain_ids, grain_oris


def mean_maxARE(y_pred, y_true):
    '''Calcuate Mean, MaxARE
    '''
    meanARE = np.around(100*np.mean((np.abs(y_pred-y_true)/y_true)), decimals=2)
    maxARE = np.around(100*np.max((np.abs(y_pred-y_true)/y_true)), decimals=2)
    return meanARE, maxARE


def seed_worker(worker_id):
    '''Seeding for DataLoaders
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(42)
    random.seed(42)


def get_data(eval, indices, b_size):
    '''Prepare data depending on Eval Case

    Parameters
    ----------
    eval : {int}
        Range 1-4, indicating type of evalution/experiment

    indices : {list}
        Indices used for splitting training/testing

    b_size : {int}
        Batch size

    Returns
    -------
    train_loader : {DataLoader}
    test_loader : {DataLoader} - For test or validation data
    '''
    g = torch.Generator()
    g.manual_seed(42)

    train_data = []
    test_data = []

    if eval == 1:
        for texture in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            with open(f'./graph_data/train/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
                # Reorder and split
                datalist = [data[i] for i in indices]
                train_data += datalist[9:]
                test_data += datalist[:9]

    elif eval == 2:
        for texture in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            with open(f'./graph_data/train/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
                train_data += data
            with open(f'./graph_data/test/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
            test_data += data

    elif eval == 3:
        for texture in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            with open(f'./graph_data/train/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
                train_data += data
        for texture in ['H', 'I', 'J', 'K', 'L']:
            with open(f'./graph_data/test/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
            test_data += data

    else:
        for texture in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            with open(f'./graph_data/train/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
            with open(f'./graph_data/test/{texture}.pickle', 'rb') as handle:
                data += pickle.load(handle)
            datalist = [data[i] for i in indices]
            train_data += datalist[:20]
        for texture in ['H', 'I', 'J', 'K', 'L']:
            with open(f'./graph_data/test/{texture}.pickle', 'rb') as handle:
                data = pickle.load(handle)
            test_data += data

    # Data Loaders
    train_loader = DataLoader(train_data, batch_size=b_size, shuffle=True, worker_init_fn=seed_worker,
                              generator=g)
    test_loader = DataLoader(
        test_data, batch_size=len(test_data), shuffle=False, worker_init_fn=seed_worker,
        generator=g)

    return train_loader, test_loader


def truncate_colormap(cmap, minval=0.0, maxval=1.0):
    '''Truncate colormap to some values

    Parameters
    ----------
    cmap : {colormap}

    minval : {float}

    maxval : {float} 

    Returns
    -------
    new_cmap : {colormap}
    '''
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, cmap.N)))
    return new_cmap


def plot_results(preds, trues, output_dir, eval, config, prop):
    '''Plot evaluation results
    '''
    sns.set(font_scale=1.75)
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    minColor = 0.4
    maxColor = 1.00
    if prop == 'strength':
        cmap = truncate_colormap(plt.get_cmap("Greens"), minColor, maxColor)
    else:
        cmap = truncate_colormap(plt.get_cmap("Blues"), minColor, maxColor)
    col = mcolors.to_hex(cmap(0.5))

    if eval != 2:
        x = np.squeeze(trues)
        y = np.squeeze(preds)
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        plt.scatter(x,
                    y,
                    c=z,
                    s=20,
                    cmap=cmap)
    else:
        plt.scatter(trues,
                    preds,
                    s=20,
                    ec='k',
                    lw=0.5,
                    color=col)
    if prop == 'strength':
        plt.xlabel('True strength (MPa)')
        plt.ylabel('Predicted strength (MPa)')
        plt.xlim([700, 1220])
        plt.ylim([700, 1220])
        plt.plot([700, 1220], [700, 1220], '-k', linewidth=2)
    else:
        plt.xlabel('True modulus (MPa)')
        plt.ylabel('Predicted modulus (MPa)')
        plt.xlim([110000, 152000])
        plt.ylim([110000, 152000])
        plt.plot([110000, 152000], [110000, 152000], '-k', linewidth=2)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    plt.savefig(f'{output_dir}/eval-{eval}_config-{config}_prop-{prop}.parity.png', dpi=300, bbox_inches="tight")


# Re-scaling Predictions
with open('./graph_data/stiffness_scaler.pickle', 'rb') as handle:
    stiffness_scaler = pickle.load(handle)
with open('./graph_data/strength_scaler.pickle', 'rb') as handle:
    strength_scaler = pickle.load(handle)
with open('./graph_data/x_scaler.pickle', 'rb') as handle:
    x_scaler = pickle.load(handle)
scaler = {}
scaler['stiffness'] = stiffness_scaler
scaler['strength'] = strength_scaler
scaler['x'] = x_scaler
