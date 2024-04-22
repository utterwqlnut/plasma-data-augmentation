from data import PlasmaDataset, generate_datasets, post_hoc_collate_fn, viewmaker_collate_fn, distort_dataset
from models import PlasmaLSTM, PlasmaViewEncoderLSTM, DecompTimeSeriesViewMaker, LSTMFormer
import os
from torch.utils.data import DataLoader
import torch
import copy
from eval import compute_metrics, plot_view, compute_metrics_after_training
from train import train_post_hoc, ViewMakerTrainer
import arg_parsing
import utils

torch.manual_seed(42)

args, arg_keys = arg_parsing.get_args()

# Create a dictionary comprehension to extract the attribute values from args
arg_values = {key: getattr(args, key) for key in arg_keys}

# Unpack the dictionary to create local variables
locals().update(arg_values)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get Dataset
file_name = 'Full_HDL_dataset_unnormalized_no_nan_column_names_w_shot.pickle'
train_dataset, test_dataset, val_dataset = generate_datasets(file_name,0.1,0.1,included_machines=included_machines, new_machine='cmod',case=case, balance=balance, device=device)

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
trainer.train(viewmaker_num_epochs)


# Generate test views
plot_view(viewmaker, test_dataset[0]['inputs_embeds'].unsqueeze(0))

# Generate Distorted Dataset
#distorted_dataset = distort_dataset(train_dataset, viewmaker, distort_d_reps, distort_nd_reps)

# Train an Post Hoc LSTM with No Augmentation
print('Beginning Post Hoc Training with No Augmentations')

train_dataloader = DataLoader(train_dataset, batch_size=post_hoc_batch_size, shuffle=False, collate_fn=post_hoc_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=post_hoc_batch_size, shuffle=False, collate_fn=post_hoc_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=post_hoc_batch_size, shuffle=False,collate_fn=post_hoc_collate_fn)


model = LSTMFormer(n_layers=1, embedding_dim=24, n_inner=48).to(device)
original_model = copy.deepcopy(model)

adam = torch.optim.Adam(params=model.parameters(),lr=post_hoc_lr)
loss_fn = torch.nn.BCELoss()

best_model = train_post_hoc(train_dataloader=train_dataloader, val_dataloader=val_dataloader, model=model, viewmaker=viewmaker, optim=adam,loss_fn=loss_fn, save_metric=post_hoc_save_metric, num_epochs=post_hoc_num_epochs, viewmaker_aug=False, max_distortion_budget=max_distortion_budget, varied_distortion_budget=varied_distortion_budget)
compute_metrics_after_training(best_model, test_dataset, prefix="No Aug ")

# Train an Post Hoc LSTM with Augmentation
print('Beginning Post Hoc Training with Augmentations')

model = original_model
adam = torch.optim.Adam(params=model.parameters(),lr=post_hoc_lr)
loss_fn = torch.nn.BCELoss()

best_model = train_post_hoc(train_dataloader=train_dataloader, val_dataloader=val_dataloader, model=model, viewmaker=viewmaker, optim=adam,loss_fn=loss_fn, save_metric=post_hoc_save_metric, num_epochs=post_hoc_num_epochs, viewmaker_aug=True, max_distortion_budget=max_distortion_budget, varied_distortion_budget=varied_distortion_budget)
compute_metrics_after_training(best_model, test_dataset, prefix="Aug ")