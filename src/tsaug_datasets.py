from data import PlasmaDataset, generate_datasets, post_hoc_collate_fn, viewmaker_collate_fn, distort_dataset
from models import PlasmaLSTM, PlasmaViewEncoderTransformer, DecompTimeSeriesViewMakerConv,DecompTimeSeriesViewMaker, LSTMFormer
import os
from torch.utils.data import DataLoader
import torch
import copy
from eval import compute_metrics, plot_view, compute_metrics_after_training
from train import train_post_hoc, ViewMakerTrainer
import arg_parsing
import utils
from utils import seed_everything
from post_hoc import compare_aug_no_aug
import pickle
import tsaug

torch.set_float32_matmul_precision('medium')


MACHINES = {"cmod": 0, "d3d": 1, "east": 2}
def save_dataset(n_reps, case, intensity):
    if case != 4:
        val_size = 0.05
    else:
        val_size = 0.1

    augmenter = (
        tsaug.TimeWarp() * 2 @ (.2 * intensity)  # time warping 5 times in parallel
        + tsaug.Drift(max_drift=(0.1, 0.3), n_drift_points=(20 * int(intensity))) @ .3  # then add Gaussian drift
        + tsaug.Quantize(n_levels=[10, 20, 30], per_channel=True) @ .4 # then quantize time series into 10/20/30 levels
        + tsaug.Dropout(size=(5), p=.1, per_channel=True) @ .4  # then drop 10 values
        + tsaug.Dropout(size=(5), fill=0.0, p=.1, per_channel=True) @ .4  # then drop 10 values
        + tsaug.AddNoise(scale=1, per_channel=True) @ .4  # then add random noise
    )

    file_name = 'Full_HDL_dataset_unnormalized_no_nan_column_names_w_shot.pickle'
    train_dataset, test_dataset, val_dataset = generate_datasets(file_name,0.1,val_size,included_machines=['cmod','d3d','east'], new_machine='cmod',case=case, balance=False)

    dataset_out = {'train': {}, 'val': {}, 'test': {}}
    train_count = 0

    for i,data in enumerate(train_dataset):

        for j in range(n_reps[MACHINES[data['machine']]]):
            dataset_out['train'][str(train_count)] = {'label': data['label'].item(), 'machine': data['machine'], 'shot': -1}
            dataset_out['train'][str(train_count)]['data'] = augmenter.augment(data['inputs_embeds'].unsqueeze(0).numpy()).squeeze()[0]
            train_count+=1

        dataset_out['train'][str(train_count)] = {'label': data['label'].item(), 'machine': data['machine'], 'shot': -1}
        dataset_out['train'][str(train_count)]['data'] = data['inputs_embeds'].detach().numpy() 
        train_count+=1
        print(f"Finished {i+1}th out of {len(train_dataset)}")

    for i,data in enumerate(val_dataset):
        dataset_out['val'][str(i)] = {'label': data['label'].item(), 'machine': data['machine'], 'shot': -1}
        dataset_out['val'][str(i)]['data'] = data['inputs_embeds'].numpy()
    for i,data in enumerate(test_dataset):
        dataset_out['test'][str(i)] = {'label': data['label'].item(), 'machine': data['machine'], 'shot': -1}
        dataset_out['test'][str(i)]['data'] = data['inputs_embeds'].numpy()
    
    print('case'+str(case)+('tsaug' if max(n_reps)>0 else '')+'.pickle') 

    with open('case'+str(case)+('tsaug' if max(n_reps)>0 else '')+'.pickle','wb') as handle:
        pickle.dump(dataset_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    del dataset_out

# MAIN

possible_reps = [(3,3,3)]
cases = [3,2]

for case in cases:
    for n_reps in possible_reps:
        save_dataset(n_reps, case, 1)