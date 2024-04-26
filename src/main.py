from data import PlasmaDataset, generate_datasets, post_hoc_collate_fn, viewmaker_collate_fn, distort_dataset
from models import PlasmaLSTM, PlasmaViewEncoderTransformer, DecompTimeSeriesViewMaker, LSTMFormer
import os
from torch.utils.data import DataLoader
import torch
import copy
from eval import compute_metrics, plot_view, compute_metrics_after_training
from train import train_post_hoc, ViewMakerTrainer
import arg_parsing
import utils
from post_hoc import compare_aug_no_aug

torch.manual_seed(42)
state = torch.get_rng_state()

args, arg_keys = arg_parsing.get_args()

# Create a dictionary comprehension to extract the attribute values from args
arg_values = {key: getattr(args, key) for key in arg_keys}

# Unpack the dictionary to create local variables
locals().update(arg_values)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get Dataset
file_name = 'Full_HDL_dataset_unnormalized_no_nan_column_names_w_shot.pickle'
train_dataset, test_dataset, val_dataset = generate_datasets(file_name,0.1,0.05,included_machines=included_machines, new_machine='cmod',case=case, balance=balance, device=device)

if viewmaker_num_steps != -1:
    viewmaker_num_epochs = viewmaker_num_steps//(len(train_dataset)//viewmaker_batch_size)

if post_hoc_num_steps != -1:
    post_hoc_num_epochs = post_hoc_num_steps//(len(train_dataset)//post_hoc_batch_size)

activations = {
    'relu': torch.nn.ReLU(),
    'sigmoid': torch.nn.Sigmoid(),
    'tanh': torch.nn.Tanh(),
}

# Train Viewmaker
viewmaker_args = {
    "n_dim": 12,
    "n_layers": viewmaker_n_layers,
    "activation": activations[viewmaker_activation],
    "default_distortion_budget": training_distortion_budget,
    "hidden_dim": viewmaker_hidden_dim,
    "layer_type": viewmaker_layer_type,
    "n_head": viewmaker_n_head,
}
encoder_args = {
    "n_input": 12,
    "n_layers": encoder_n_layers,
    "h_size": encoder_hidden_dim,
    "out_size": encoder_out_size,
}


viewmaker = DecompTimeSeriesViewMaker(**viewmaker_args).to(device)

encoder = PlasmaViewEncoderTransformer(**encoder_args).to(device)


viewmaker_trainer_args = {
    "batch_size": viewmaker_batch_size,
    "t": viewmaker_loss_t,
    "v_loss_weight": viewmaker_loss_weight,
    "v_lr": v_lr,
    "e_lr": e_lr,
    "viewmaker": viewmaker,
    "encoder": encoder,
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    "collate_fn": viewmaker_collate_fn,
}

utils.set_up_wandb(
        training_args=viewmaker_trainer_args, seed=42, parsed_args=arg_values)

trainer = ViewMakerTrainer(**viewmaker_trainer_args)
trainer.train(viewmaker_num_epochs)

# Generate test views
plot_view(viewmaker, test_dataset[0]['inputs_embeds'][:-test_dataset.cutoff_steps].unsqueeze(0))

# Generate Distorted Dataset

compare_aug_no_aug(train_dataset=train_dataset, 
                   test_dataset=test_dataset, 
                   val_dataset=val_dataset,
                   post_hoc_batch_size=post_hoc_batch_size,
                   post_hoc_lr=post_hoc_lr,
                   post_hoc_num_epochs=post_hoc_num_epochs,
                   post_hoc_save_metric=post_hoc_save_metric,
                   viewmaker=viewmaker,
                   distort_d_reps=distort_d_reps,
                   distort_nd_reps=distort_nd_reps,
                   state=state,
                   device=device)
