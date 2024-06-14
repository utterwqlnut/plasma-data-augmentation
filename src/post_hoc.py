from data import PlasmaDataset, generate_datasets, post_hoc_collate_fn, viewmaker_collate_fn, distort_dataset, BatchSampler
from models import PlasmaLSTM, PlasmaViewEncoderLSTM, DecompTimeSeriesViewMaker, LSTMFormer
import os
from torch.utils.data import DataLoader
import torch
import copy
from eval import compute_metrics, plot_view, compute_metrics_after_training
from train import train_post_hoc, ViewMakerTrainer
import arg_parsing
import utils
from utils import seed_everything


def compare_aug_no_aug(
    train_dataset,
    test_dataset,
    val_dataset,
    post_hoc_batch_size,
    post_hoc_lr,
    post_hoc_save_metric,
    post_hoc_num_epochs,
    viewmaker,
    distort_d_reps,
    distort_nd_reps,
    state,
    fabric,
    train_ecs=3,
):

    # Train an Post Hoc LSTM with No Augmentation
    print('Beginning Post Hoc Training with No Augmentations')

    # Set state so runs are same accross different viewmaker changes
    seed_everything(42)

    windowed_train = copy.deepcopy(train_dataset)

    windowed_train.window(True, train_ecs)
    val_dataset.window(False,1)
    test_dataset.window(False,1)

    train_lengths = []
    val_lengths = []

    for data in windowed_train:
        train_lengths.append(len(data['inputs_embeds']))

    for data in val_dataset:
        val_lengths.append(len(data['inputs_embeds']))

    train_dataloader = DataLoader(windowed_train, batch_sampler=BatchSampler(train_lengths,post_hoc_batch_size), shuffle=False, collate_fn=post_hoc_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_sampler=BatchSampler(val_lengths,post_hoc_batch_size), shuffle=False,collate_fn=post_hoc_collate_fn)

    # Simplified version of moddel for testing, change when running
    model = LSTMFormer()
    original_model = copy.deepcopy(model)

    seed_everything(42)

    adam = torch.optim.Adam(params=model.parameters(),lr=post_hoc_lr)
    loss_fn = torch.nn.BCELoss()

    best_model = train_post_hoc(train_dataloader=train_dataloader, val_dataloader=val_dataloader,val_dataset=val_dataset, model=model, viewmaker=viewmaker, optim=adam,loss_fn=loss_fn, save_metric=post_hoc_save_metric, num_epochs=post_hoc_num_epochs, viewmaker_aug=False, fabric=fabric)
    compute_metrics_after_training(best_model, test_dataset, fabric.device, prefix="No Aug ")

    # Distort dataset
    distort_dataset(train_dataset, viewmaker, distort_d_reps, distort_nd_reps, fabric.device)
    windowed_train = copy.deepcopy(train_dataset)
    windowed_train.window(True,train_ecs)

    train_lengths = []
    for data in windowed_train:
        train_lengths.append(len(data['inputs_embeds']))

    train_dataloader = DataLoader(windowed_train, batch_sampler=BatchSampler(train_lengths,post_hoc_batch_size), shuffle=False, collate_fn=post_hoc_collate_fn)

    # Train an Post Hoc LSTM with Augmentation
    print('Beginning Post Hoc Training with Augmentations')

    model = original_model
    seed_everything(42)
    adam = torch.optim.Adam(params=model.parameters(),lr=post_hoc_lr)
    loss_fn = torch.nn.BCELoss()

    best_model = train_post_hoc(train_dataloader=train_dataloader, val_dataloader=val_dataloader,val_dataset=val_dataset, model=model, viewmaker=viewmaker, optim=adam,loss_fn=loss_fn, save_metric=post_hoc_save_metric, num_epochs=post_hoc_num_epochs, viewmaker_aug=True, fabric=fabric)
    compute_metrics_after_training(best_model, test_dataset,fabric.device, prefix="Aug ")