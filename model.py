import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from torch_geometric.nn import Linear, SAGEConv, global_mean_pool
import time
from utils import *


def seed_all(seed):
    '''
    Set random seeds for reproducability
    '''
    if not seed:
        seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GNN(torch.nn.Module):
    '''
    Graph Neural Network
    '''
    def __init__(self, N_fl1, N_mpl, N_fl2, N_fl3):
        super(GNN, self).__init__()
        self.pre = Linear(5, N_fl1)
        self.conv1 = SAGEConv(N_fl1, N_mpl, normalize=True)
        self.conv2 = SAGEConv(N_mpl, N_mpl, normalize=True)
        self.post1 = Linear(N_mpl, N_fl2)
        self.post2 = Linear(N_fl2, N_fl3)
        self.out = Linear(N_fl3, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Pre Processing Linear Layer
        x = F.relu(self.pre(x))
        # 1. Obtain node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # 2. Readout layer
        x = global_mean_pool(x, batch)
        # 3. Apply Fully Connected Layers
        x = F.relu(self.post1(x))
        x = F.relu(self.post2(x))
        x = self.out(x)
        return x


def init_model():
    '''
    Initialize model
    '''
    seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(N_fl1, N_mpl, N_fl2, N_fl3).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=l_rate, weight_decay=w_decay)
    return model, optimizer


def train(model, optimizer, train_loader, val_loader, n_epoch, prop, config, fold):
    '''
    Train GNN
    '''
    filename = f'{output_dir}/eval-{eval}_config-{config}_fold-{fold}_loss_history.txt'
    output = open(filename, "w")

    print('Epoch Training_MSE Validation_MSE', file=output, flush=True)

    seed_all(seed)
    for epoch in range(n_epoch):
        model.train()
        # Train batches
        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            train_pred = model(train_batch)
            train_true = getattr(train_batch, prop)
            train_loss = F.mse_loss(train_pred, train_true)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Evaluate
        val_pred, val_true = test(model, val_loader, prop)
        val_loss = F.mse_loss(val_pred, val_true)
        print(f'{epoch:d}, {train_loss:e}, {val_loss:e}', file=output, flush=True)
    return


def test(model, data_loader, prop):
    '''
    Test GNN
    '''
    seed_all(seed)
    model.eval()
    data = next(iter(data_loader)).to(device)
    pred = model(data)
    true = getattr(data, prop)
    return pred, true


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=int, default=1)
    parser.add_argument('--prop', type=str, default='stiffness')
    parser.add_argument('--config_dir', type=str, default='./config/')
    parser.add_argument('--config', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./output/')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    eval = args.eval
    prop = args.prop
    config_dir = args.config_dir
    config = args.config
    output_dir = args.output_dir
    seed = args.seed

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config_name = config_dir + str(config) + '.json'
    with open(config_name, 'r') as h:
        params = json.load(h)

    l_rate = params['l_rate']
    w_decay = params['w_decay']
    n_epoch = params['n_epoch']
    b_size = params['b_size']
    N_fl1 = params['N_fl1']
    N_mpl = params['N_mpl']
    N_fl2 = params['N_fl2']
    N_fl3 = params['N_fl3']

    # Set seeds for complete reproducability
    seed_all(seed)

    # Define the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cases = ['10-fold CV (A-G)',
             'Train (A-G) / Test (A-G)',
             'Train (A-G) / Test (H-L)',
             '5-fold reduced Train (A-G) / Test (H-L)']

    print('\n====== Configuration ======')
    print(f'Evaluation #{eval}:\t\t{cases[eval-1]}')
    print(f'Regression task:\t{prop}')
    print(f'Hyper-parameters :\t{config}.json')

# *************************************************************************** #
    print('\n====== Training / Testing ======')
    start = time.time()

    # Eval == 1 or 4 : 10-fold CV (A-G) or 5-fold Train Test (A-G), (H-L)
    if eval in [1, 4]:
        preds = torch.empty((0, 1)).to(device)
        trues = torch.empty((0, 1)).to(device)

        # 10-fold on 90 RVEs from Train/Val Set
        if eval == 1:
            kfold = np.random.choice(90, [10, 9], replace=False)
            num_folds = 10

        # 5 trainings on 20 RVEs, Entire A-G set considered
        else:
            kfold = np.random.choice(100, [5, 20], replace=False)
            num_folds = 5

        for k in range(num_folds):
            print('fold-{}'.format(k+1), end='')
            kfold = np.concatenate([kfold[1:, :], [kfold[0, :]]])
            kfold_ids = kfold.flatten()

            # Load data
            train_loader, val_loader = get_data(eval, kfold_ids, b_size)

            # Define model and optimizer
            model, optimizer = init_model()

            # Train model
            train(model, optimizer, train_loader,
                  val_loader, n_epoch, prop, config, k)

            # Test model
            k_pred, k_true = test(model, val_loader, prop)

            # Record predictions
            preds = torch.cat([preds, k_pred])
            trues = torch.cat([trues, k_true])

            # Save model
            torch.save(
                model, f'{output_dir}/eval-{eval}_config-{config}_prop-{prop}_fold-{k+1}_checkpoint.pth')
            print('\t\tcompleted')

# *************************************************************************** #
    # Eval == 2 or 3 : Train/Test (A-G) or Train/Test (A-G)/(H-L)
    else:
        # Load data
        train_loader, test_loader = get_data(eval, [], b_size)

        # Define model and optimizer
        model, optimizer = init_model()

        # Train model
        train(model, optimizer, train_loader, test_loader,
              n_epoch, prop, config, 'NA')

        # Test Model
        preds, trues = test(model, test_loader, prop)

        # Save model
        torch.save(
            model, f"{output_dir}/eval-{eval}_config-{config}_prop-{prop}_fold-{'NA'}_checkpoint.pth")

    print(f'Processing time: {time.time()-start:.2f} seconds')
# *************************************************************************** #
    # Report and Visualize predictions

    print('\n====== RESULTS ======')
    preds = scaler[prop].inverse_transform(
        preds.detach().detach().cpu().numpy())
    trues = scaler[prop].inverse_transform(
        trues.detach().detach().cpu().numpy())
    meanARE, maxARE = mean_maxARE(preds, trues)

    print(f'(MeanARE, MaxARE):\t({meanARE}, {maxARE})')
    plot_results(preds, trues, output_dir, eval, config, prop)
