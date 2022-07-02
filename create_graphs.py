import networkx as nx
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from utils import *

textures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
data_dir = './orientation_data/'

for texture in tqdm(textures):

    # Load RVE orientation data
    ori_data = sio.loadmat(f'{data_dir}rve_quaternions_{texture}.mat')['quaternions']
    ori_micro = ori_data.reshape((100, ori_data.shape[1], 21, 21, 21))

    # Loop through RVEs
    for i in range(100):
        grain_ids, grain_oris = assign_grains(ori_micro[i, :, :, :, :])
        nbr_dict = get_nbrs(grain_ids, periodic=True)

        # Create graph
        G = nx.Graph()
        for feat, ori in enumerate(grain_oris):
            size = np.sum(grain_ids == feat+1)
            G.add_nodes_from(
                [(feat+1, {"x": np.hstack([ori, size])})]
            )
        for feat, (nbrs, areas) in nbr_dict.items():
            for nbr, area in zip(nbrs, areas):
                G.add_edge(feat, nbr, edge_weight=area)
        nx.write_gpickle(
            G, f'./graphs/{texture}/rve_{i}')
