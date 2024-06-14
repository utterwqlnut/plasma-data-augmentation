from data import PlasmaDataset, generate_datasets, post_hoc_collate_fn, viewmaker_collate_fn, distort_dataset
from models import TestViewmaker, PlasmaLSTM, PlasmaViewEncoderTransformer, DecompTimeSeriesViewMakerConv,DecompTimeSeriesViewMaker, LSTMFormer
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
from lightning import Fabric

torch.set_float32_matmul_precision('medium')

fabric = Fabric(accelerator = "cuda" if torch.cuda.is_available() else "cpu", devices=1, precision="16-mixed" if torch.cuda.is_available() else "bf16-mixed")
fabric.launch()

MACHINES = {"cmod": 0, "d3d": 1, "east": 2}
def save_dataset(n_reps, case):
    if case != 4:
        val_size = 0.05
    else:
        val_size = 0.1

    viewmaker_args = {
        "n_dim": 12,
        "n_layers": 4,
        "activation": torch.nn.GELU(),
        "default_distortion_budget": 0.1,
        "hidden_dim":128,
        "layer_type": 'transformer',
        "n_head": 1,
    }

    viewmaker = TestViewmaker(**viewmaker_args)
    state_dict = torch.load('viewmaker'+str(case)+'.pt', map_location='cpu')
    viewmaker.load_state_dict(state_dict,strict=False)

    viewmaker = fabric.setup(viewmaker)
    
    file_name = 'Full_HDL_dataset_unnormalized_no_nan_column_names_w_shot.pickle'
    train_dataset, test_dataset, val_dataset = generate_datasets(file_name,0.1,val_size,included_machines=['cmod','d3d','east'], new_machine='cmod',case=case, balance=False)

    dataset_out = {'train': {}, 'val': {}, 'test': {}}
    train_count = 0

    for i,data in enumerate(train_dataset):

        for j in range(n_reps[MACHINES[data['machine']]]):
            dataset_out['train'][str(train_count)] = {'label': data['label'].item(), 'machine': data['machine'], 'shot': -1}
            dataset_out['train'][str(train_count)]['data'] = viewmaker(data['inputs_embeds'].unsqueeze(0).to(fabric.device)).squeeze().detach().cpu().numpy() 
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
    
    print('case'+str(case)+('aug' if max(n_reps)>0 else '')+'.pickle') 

    with open('case'+str(case)+('aug' if max(n_reps)>0 else '')+'.pickle','wb') as handle:
        pickle.dump(dataset_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    del dataset_out

# MAIN

possible_reps = [(0,0,0)]
cases = [3]

for case in cases:
    for n_reps in possible_reps:
        save_dataset(n_reps, case)