import random

import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

from fastdtw import fastdtw
from sklearn.linear_model import Ridge

import arg_parsing
from models import PlasmaLSTM
from data import generate_datasets

import warnings
warnings.filterwarnings('once') 

#############
# Functions #
#############

def get_d_nd_shots(model, dataset, balance=False, seed=None):
    nd_shots, d_shots = [], []
    for shot in dataset:
        input = shot['inputs_embeds'][:-dataset.cutoff_steps]
        out = model(input.unsqueeze(0).to(torch.float32)).squeeze()
        y_pred = np.round(out.cpu().detach().numpy())

        d_shots.append(input) if y_pred else nd_shots.append(input)
    
    if balance:
        random.seed(seed)
        num_samples = min(len(d_shots), len(nd_shots))
        nd_shots, d_shots = random.sample(nd_shots, num_samples), random.sample(d_shots, num_samples)

    return nd_shots, d_shots


def get_z_from_zprime(x, z_prime):
    noise = torch.zeros(x.size())
    noised_cols = np.where(z_prime == 0)[0]

    for col in noised_cols:
        std = torch.std(x[:, col])
        noise[:, col] = std * torch.randn(len(x))
    z = noise + x

    return z

def lime(model, x, y, num_samples=120, kernel=lambda d: np.sqrt(np.exp(-(d ** 2)))):
    '''
    The LIME algorithm
    Args:
        model: 
        x (torch.tensor): Original data tensor to explain
        y (int): Prediction for x
        num_samples (int): Number of points to sample from x
        kernel (func): Function weighting distances between points
    Returns
        coefs (list[tuple]): List of tuples (feature column number. regression coefficient)
    '''

    # Generate interpretable representation of x
    x_prime = np.ones(x.size(1))

    # Generate neighborhood of random samples around x'
    Z_prime = []
    Z = []
    for i in range(num_samples): 

        z_prime = np.random.randint(2, size=len(x_prime))
        Z_prime.append(z_prime)

        z = get_z_from_zprime(x, z_prime)
        Z.append(z)
    
    # LIME algorithm
    y_lime = [y]
    X_lime = [x_prime] 
    distances = [0]  # weights for each data instance in regression model

    for i, z in enumerate(Z):

        d = fastdtw(x, z)[0]  # LIME for univariate time series paper suggest DTW as distance metric
        distances.append(d)
        
        input = z.unsqueeze(0).to(torch.float32)
        out = model(input).squeeze()
        y_i = round(out.item())

        y_lime.append(y_i)
        X_lime.append(Z_prime[i])
    
    distances = np.array(distances)
    X_lime = np.array(X_lime)
    y_lime = np.array(y_lime)

    distances = (distances - np.mean(distances)) / np.std(distances)
    weights = kernel(distances)

    clf = Ridge(alpha=0.01) # value used in LIME repo
    clf.fit(X_lime, y_lime, weights)
    coefs = clf.coef_
    coefs = sorted(zip(list(range(len(coefs))), coefs), key=lambda x: np.abs(x[1]), reverse=True)

    return coefs

def lime_sweep(model, feats, embeds, num_samples=120, kernel=lambda d: np.sqrt(np.exp(-(d ** 2))), label=None):

    feat_counts = [{'feat': feat, 'count': 0, 'weight': 0} for feat in feats]

    for embed in embeds:

        if label is None:   # if (for whatever reason), the sweep isn't over the same label
            input = embed.unsqueeze(0).to(torch.float32)
            out = model(input).squeeze()
            y_pred = round(out.item())
        else:
            y_pred = label

        coefs = lime(model, embed, y_pred, num_samples, kernel)

        # ++ the most important feature's count
        i_max = coefs[0][0]
        feat_counts[i_max]['count'] += 1

        # add the absolute + normalized regression weights
        tot_weight = sum(abs(coef) for _, coef in coefs)
        if tot_weight == 0:
            tot_weight = 1    # don't divide by 0
        
        for ind, coef in coefs:
            feat_counts[ind]['weight'] += abs(coef) / tot_weight
    
    # Log important metrics on wandb
    title = f'LIME Sweep for Class {label}'
    fig, axs = plt.subplots(2)
    labels = [feat['feat'] for feat in feat_counts]
    y_pos = list(range(len(labels)))

    freq = [feat['count'] / len(embeds) for feat in feat_counts]
    hbars0 = axs[0].barh(y_pos, freq, align='center')
    axs[0].set_yticks(y_pos, labels=labels)
    axs[0].invert_yaxis()  # labels read top-to-bottom
    axs[0].set_xlabel('Percentage of Runs as Most Important')
    axs[0].bar_label(hbars0, fmt='%.2f')  # Label with specially formatted floats
    axs[0].set_xlim(right=1.2)  # adjust xlim to fit labels

    avg_weight = [feat['weight'] / len(embeds) for feat in feat_counts]
    hbars1 = axs[1].barh(y_pos, avg_weight, align='center')
    axs[1].set_yticks(y_pos, labels=labels)
    axs[1].invert_yaxis()
    axs[1].set_xlabel('Average (Normalized) Regression Weight')
    # TODO: not sure if this is the best metric when regression weights are usually 0
    axs[1].bar_label(hbars1, fmt='%.2f')
    axs[1].set_xlim(right=1.2)

    fig.tight_layout()
    wandb.log({title:wandb.Image(fig)})

########
# Main #
########

torch.manual_seed(42)
np.random.seed(42)

# Add args as local variables
args, arg_keys = arg_parsing.get_args()
arg_values = {key: getattr(args, key) for key in arg_keys}
locals().update(arg_values)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get Dataset
file_name = 'Full_HDL_dataset_unnormalized_no_nan_column_names_w_shot.pickle'
train_dataset, test_dataset, val_dataset = generate_datasets(file_name,0.1,0.1,included_machines=included_machines, new_machine='cmod',case=case, balance=balance, device=device)

# Load previously saved model
lstm_path = '../models/toy_lstm.pt'
aug_lstm_path = '../models/toy_lstm_aug.pt'

lstm = PlasmaLSTM(12,post_hoc_n_layers,post_hoc_h_size).to(device)
lstm.load_state_dict(torch.load(lstm_path))
lstm.eval()

'''lstm_aug = PlasmaLSTM(12,post_hoc_n_layers,post_hoc_h_size).to(device)
lstm_aug.load_state_dict(torch.load(aug_lstm_path))
lstm_aug.eval()'''

wandb.init(project="lime-sweep",
               entity="autoformer-xai")

feats = ['kappa', 'q95', 'li', 'ip_error_normalized', 'n_equal_1_normalized',
       'd3d', 'v_loop', 'lower_gap', 'cmod', 'Greenwald_fraction', 'beta_p',
       'east']

# Get d and nd_shots
nd_shots, d_shots = get_d_nd_shots(lstm, test_dataset, balance=True, seed=42)

# Sweep over nd shots
lime_sweep(lstm, feats, nd_shots, label=0)

# Sweep over d shots
lime_sweep(lstm, feats, d_shots, label=1)
