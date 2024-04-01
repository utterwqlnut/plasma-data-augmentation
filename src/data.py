import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import sys
import os
import pickle
from sklearn.preprocessing import RobustScaler
import random

class PlasmaDataset(Dataset):

    def __init__(self, shot_data, device, cutoff_steps=4):
        self.cutoff_steps = cutoff_steps
        self.labels = []
        self.machines = []
        self.inputs_embeds = []
        self.predict_inputs_embeds = []

        for i in range(len(shot_data)):
            item = shot_data[i]
            if len(item['data'])<=cutoff_steps:
                continue

            self.labels.append(torch.tensor(item['label']).to(device))
            self.machines.append(item['machine'])
            self.inputs_embeds.append(torch.tensor(item['data'].values,dtype=torch.float32).to(device))
            self.predict_inputs_embeds.append(torch.tensor(item['data'].values[:-cutoff_steps],dtype=torch.float32).to(device))


    def __getitem__(self, index):
        return {'label':self.labels[index],
                'machine': self.machines[index], 
                'inputs_embeds': self.inputs_embeds[index], 
                'predict_inputs_embeds': self.predict_inputs_embeds[index]}
    
    def __len__(self):
        return len(self.labels)
    
    def load_file(file_name):
        file = open(file_name,'rb')
        data = pickle.load(file)

        return data

def generate_datasets(file_name: str, train_size: float, test_size: float, val_size: float, device, included_machines=['cmod','d3d','east'], balance=False):
    data = PlasmaDataset.load_file(os.path.dirname(__file__)+'/../data/'+file_name)

    # Convert to list of dicts
    shot_data = []
    for shot in data.values():
        if shot['machine'] in included_machines:
            shot_data.append(shot)

    random.shuffle(shot_data)
    
    # Get datasets
    if not balance:
        train_dataset = PlasmaDataset(
            shot_data=shot_data[:int(train_size*len(shot_data))],
            device=device
        )
    else:
        balanced_data = balance_data(shot_data[:int(train_size*len(shot_data))])
        train_dataset = PlasmaDataset(
            shot_data=balanced_data,
            device=device
        )
    test_dataset = PlasmaDataset(
        shot_data=shot_data[len(train_dataset):len(train_dataset)+int(test_size*len(shot_data))],
        device=device
    )
    val_dataset = PlasmaDataset(
        shot_data=shot_data[len(test_dataset):],
        device=device
    )

    return train_dataset, test_dataset, val_dataset

def balance_data(shot_data):
    total_count = len(shot_data)
    d_count = sum([shot_data[i]['label'] for i in range(total_count)])
    nd_counter = 0

    new_data = []

    for i in range(len(shot_data)):
        if nd_counter<d_count and shot_data[i]['label']==0:
            nd_counter+=1
            new_data.append(shot_data[i])
        elif shot_data[i]['label']==0:
            continue
        else:
            new_data.append(shot_data[i])

    return new_data

   
def post_hoc_collate_fn(dataset):
    output = {}

    output['inputs_embeds'] = pad_sequence(
        [df["predict_inputs_embeds"].to(dtype=torch.float32) for df in dataset],
        padding_value=-100,
        batch_first=True)
    output['labels'] = torch.stack([df['label'].to(dtype=torch.float32) for df in dataset]).unsqueeze(-1)

    return output

def viewmaker_collate_fn(dataset):
    output = {}

    output['inputs_embeds'] = pad_sequence(
        [df["inputs_embeds"].to(dtype=torch.float32) for df in dataset],
        padding_value=-100,
        batch_first=True)
    output['labels'] = torch.stack([df['label'].to(dtype=torch.float32) for df in dataset]).unsqueeze(-1)

    return output

def distort_dataset(dataset, model, d_reps, nd_reps):
    new_predict_inputs_embeds = []
    new_inputs_embeds = []
    new_labels = []
    new_machines = []
    for i, data in enumerate(dataset.predict_inputs_embeds):
        if dataset.labels[i] == 1:
            for j in range(d_reps):
                new_predict_inputs_embeds.append(model(data.unsqueeze(0)).squeeze())
                new_inputs_embeds.append(model(dataset.inputs_embeds[i].unsqueeze(0)).squeeze())
                new_labels.append(dataset.labels[i])
                new_machines.append(dataset.machines[i])
                
        else:
            for j in range(nd_reps):
                new_predict_inputs_embeds.append(model(data.unsqueeze(0)).squeeze())
                new_inputs_embeds.append(model(dataset.inputs_embeds[i].unsqueeze(0)).squeeze())
                new_labels.append(dataset.labels[i])
                new_machines.append(dataset.machines[i])
    
    dataset.predict_inputs_embeds = new_predict_inputs_embeds
    dataset.inputs_embeds = new_inputs_embeds
    dataset.labels = new_labels
    dataset.machines = new_machines
