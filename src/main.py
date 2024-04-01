from data import PlasmaDataset, generate_datasets, post_hoc_collate_fn, viewmaker_collate_fn, distort_dataset
from models import PlasmaLSTM, PlasmaViewEncoderLSTM, TimeSeriesViewMaker
import os
from torch.utils.data import DataLoader
import torch
import copy
from eval import compute_metrics, plot_view, compute_metrics_after_training
from train import train_post_hoc, ViewMakerTrainer
import arg_parsing
import utils

args, arg_keys = arg_parsing.get_args()

# Create a dictionary comprehension to extract the attribute values from args
arg_values = {key: getattr(args, key) for key in arg_keys}

# Unpack the dictionary to create local variables
locals().update(arg_values)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get Dataset
file_name = 'Full_HDL_dataset_unnormalized_no_nan_column_names_w_shot.pickle'
train_dataset, test_dataset, val_dataset = generate_datasets(file_name,0.8,0.1,0.1,included_machines=included_machines, balance=balance, device=device)

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
    "distortion_budget": viewmaker_distortion_budget,
    "hidden_dim": viewmaker_hidden_dim,
    "layer_type": viewmaker_layer_type,
    "n_head": viewmaker_n_head,
}
encoder_args = {
    "n_input": 12,
    "n_layers": encoder_n_layers,
    "h_size": encoder_hidden_dim,
}


viewmaker = TimeSeriesViewMaker(**viewmaker_args).to(device)

encoder = PlasmaViewEncoderLSTM(**encoder_args).to(device)


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
#trainer.train(viewmaker_num_epochs)

# Generate test views
plot_view(viewmaker, test_dataset[0]['inputs_embeds'].unsqueeze(0))

# Generate Distorted Dataset
distort_dataset(train_dataset, viewmaker, distort_d_reps, distort_nd_reps)

# Train an Post Hoc LSTM
train_dataloader = DataLoader(train_dataset, batch_size=post_hoc_batch_size, shuffle=True, collate_fn=post_hoc_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=post_hoc_batch_size, shuffle=True, collate_fn=post_hoc_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=post_hoc_batch_size, shuffle=True,collate_fn=post_hoc_collate_fn)


model = PlasmaLSTM(12,post_hoc_n_layers,post_hoc_h_size).to(device)
adam = torch.optim.Adam(params=model.parameters(),lr=post_hoc_lr)
loss_fn = torch.nn.BCELoss()
save_metric = 'accuracy'

best_model = train_post_hoc(train_dataloader=train_dataloader, val_dataloader=test_dataloader, model=model,optim=adam,loss_fn=loss_fn, save_metric=save_metric, num_epochs=post_hoc_num_epochs)
compute_metrics_after_training(best_model, test_dataset)