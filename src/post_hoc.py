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


def compare_aug_no_aug(
    train_dataset,
    distorted_dataset,
    test_dataset,
    val_dataset,
    post_hoc_batch_size,
    post_hoc_lr,
    post_hoc_save_metric,
    post_hoc_num_epochs,
    viewmaker,
    state,
    device,
):
    # Train an Post Hoc LSTM with No Augmentation
    print('Beginning Post Hoc Training with No Augmentations')
    torch.set_rng_state(state)
    

    train_dataloader = DataLoader(train_dataset, batch_size=post_hoc_batch_size, shuffle=False, collate_fn=post_hoc_collate_fn)
    distorted_dataloader = DataLoader(distorted_dataset, batch_size=post_hoc_batch_size, shuffle=False, collate_fn=post_hoc_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=post_hoc_batch_size, shuffle=False,collate_fn=post_hoc_collate_fn)

    # Simplified version of moddel
    model = LSTMFormer(n_layers=1, embedding_dim=24, n_inner=48).to(device)
    original_model = copy.deepcopy(model)

    adam = torch.optim.Adam(params=model.parameters(),lr=post_hoc_lr)
    loss_fn = torch.nn.BCELoss()

    best_model = train_post_hoc(train_dataloader=train_dataloader, val_dataloader=val_dataloader,val_dataset=val_dataset, model=model, viewmaker=viewmaker, optim=adam,loss_fn=loss_fn, save_metric=post_hoc_save_metric, num_epochs=post_hoc_num_epochs, viewmaker_aug=False)
    compute_metrics_after_training(best_model, test_dataset, prefix="No Aug ")

    # Train an Post Hoc LSTM with Augmentation
    print('Beginning Post Hoc Training with Augmentations')
    torch.set_rng_state(state)

    model = original_model
    adam = torch.optim.Adam(params=model.parameters(),lr=post_hoc_lr)
    loss_fn = torch.nn.BCELoss()

    best_model = train_post_hoc(train_dataloader=distorted_dataloader, val_dataloader=val_dataloader,val_dataset=val_dataset, model=model, viewmaker=viewmaker, optim=adam,loss_fn=loss_fn, save_metric=post_hoc_save_metric, num_epochs=post_hoc_num_epochs, viewmaker_aug=True)
    compute_metrics_after_training(best_model, test_dataset, prefix="Aug ")