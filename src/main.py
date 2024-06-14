from data import PlasmaDataset, generate_datasets, post_hoc_collate_fn, viewmaker_collate_fn, distort_dataset
from models import TestViewmaker, PlasmaLSTM, PlasmaViewEncoderTransformer, DecompTimeSeriesViewMaker, DecompTimeSeriesViewMakerConv, LSTMFormer, DisruptMLPHead
import os
from torch.utils.data import DataLoader
import torch
import copy
from eval import compute_metrics, plot_view, compute_metrics_after_training
from train import train_post_hoc, ViewMakerTrainer, ViewMakerPreTrainer
import arg_parsing
import utils
from utils import seed_everything
from post_hoc import compare_aug_no_aug
from lightning import Fabric

torch.set_float32_matmul_precision('medium')

state = torch.get_rng_state()

args, arg_keys = arg_parsing.get_args()

# Create a dictionary comprehension to extract the attribute values from args
arg_values = {key: getattr(args, key) for key in arg_keys}

# Unpack the dictionary to create local variables
locals().update(arg_values)

utils.set_up_wandb(seed=42, parsed_args=arg_values)

# Change to specifications of computer
fabric = Fabric(accelerator = "cuda" if torch.cuda.is_available() else "cpu", devices=1, precision="16-mixed" if torch.cuda.is_available() else "bf16-mixed")
fabric.launch()

seed_everything(42)

# Get Dataset
file_name = 'Full_HDL_dataset_unnormalized_no_nan_column_names_w_shot.pickle'

if case != 4:
    val_size = 0.05
else:
    val_size = 0.1

train_dataset, test_dataset, val_dataset = generate_datasets(file_name,0.1,val_size,included_machines=included_machines, new_machine='cmod',case=case, balance=balance)

if viewmaker_num_steps!=-1:
    viewmaker_num_epochs = (viewmaker_num_steps)//(len(train_dataset)//viewmaker_batch_size)

if viewmaker_num_pretrain_steps!=-1:
    viewmaker_num_pretrain_epochs = (viewmaker_num_pretrain_steps)//(len(train_dataset)//viewmaker_batch_size)

activations = {
    'relu': torch.nn.ReLU(),
    'sigmoid': torch.nn.Sigmoid(),
    'tanh': torch.nn.Tanh(),
    'leaky': torch.nn.LeakyReLU(),
    'gelu': torch.nn.GELU(),
}

# Setup Viewmaker and Encoder
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

mlp_args = {
    "n_dim": encoder_out_size,
    "hidden_dim": encoder_out_size*2,
    "activation": activations[viewmaker_activation]
}

viewmaker = TestViewmaker(**viewmaker_args)

encoder = PlasmaViewEncoderTransformer(**encoder_args)

mlp = DisruptMLPHead(**mlp_args)

viewmaker_pretrainer_args = {
    "batch_size": viewmaker_batch_size,
    "v_lr": v_lr,
    "viewmaker": viewmaker,
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    "collate_fn": viewmaker_collate_fn,
    "fabric": fabric,
}
viewmaker_trainer_args = {
    "batch_size": viewmaker_batch_size,
    "t": viewmaker_loss_t,
    "v_loss_weight": viewmaker_loss_weight,
    "v_lr": v_lr,
    "e_lr": e_lr,
    "m_lr": m_lr,
    "viewmaker": viewmaker,
    "encoder": encoder,
    "mlp": mlp,
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    "collate_fn": viewmaker_collate_fn,
    "fabric": fabric,
}


# Pretrain Viewmaker
pretrainer = ViewMakerPreTrainer(**viewmaker_pretrainer_args)
#pretrainer.train(viewmaker_num_pretrain_epochs)

# Generate test view after pretraining
plot_view(viewmaker, test_dataset[0]['inputs_embeds'][:-test_dataset.cutoff_steps].unsqueeze(0).to(fabric.device))

# Train Viewmaker
trainer = ViewMakerTrainer(**viewmaker_trainer_args)
trainer.train(viewmaker_num_epochs)

torch.save(viewmaker.state_dict(), 'viewmaker'+str(case)+'.pt')

# Generate test views
plot_view(viewmaker, test_dataset[0]['inputs_embeds'][:-test_dataset.cutoff_steps].unsqueeze(0).to(fabric.device))

# Compare post hoc models
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
                   fabric=fabric,)