import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import sys
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import random
import copy

random.seed(24)

class PlasmaDataset(Dataset):

    def __init__(self, shot_data, device, cutoff_steps=4):
        self.cutoff_steps = cutoff_steps
        self.labels = []
        self.machines = []
        self.inputs_embeds = []
        self.predict_inputs_embeds = []
        self.device = device

        for i in range(len(shot_data)):
            item = shot_data[i]
            if len(item['data'])<=cutoff_steps:
                continue

            self.labels.append(torch.tensor(item['label']))
            self.machines.append(item['machine'])
            self.inputs_embeds.append(torch.tensor(item['data'].values,dtype=torch.float32))
            self.predict_inputs_embeds.append(torch.tensor(item['data'].values[:-cutoff_steps],dtype=torch.float32))


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

    def scale(self):
        combined_full = torch.cat(self.inputs_embeds)
        combined_predict = torch.cat(self.predict_inputs_embeds)

        scaler = StandardScaler().fit(combined_full)

        with torch.no_grad():
            for i in range(len(self.inputs_embeds)):
                self.inputs_embeds[i] = torch.tensor(normalize(scaler.transform(self.inputs_embeds[i])),dtype=torch.float32)
                self.predict_inputs_embeds[i] = torch.tensor(normalize(scaler.transform(self.predict_inputs_embeds[i])),dtype=torch.float32)

        return scaler
    
    def scale_w_scaler(self, scaler):
        with torch.no_grad():
            for i in range(len(self.inputs_embeds)):
                self.inputs_embeds[i] = torch.tensor(normalize(scaler.transform(self.inputs_embeds[i])),dtype=torch.float32)
                self.predict_inputs_embeds[i] = torch.tensor(normalize(scaler.transform(self.predict_inputs_embeds[i])),dtype=torch.float32)

    def move_to_device(self):
        for i in range(len(self.inputs_embeds)):
            self.inputs_embeds[i]=self.inputs_embeds[i].to(self.device)
            self.predict_inputs_embeds[i]=self.predict_inputs_embeds[i].to(self.device)
            self.labels[i]=self.labels[i].to(self.device)

    def get_machine_counts(self):
        cmod_count = 0
        d3d_count = 0
        east_count = 0

        for i in range(len(self.machines)):
            if self.machines[i] == 'cmod':
                cmod_count+=1
            elif self.machines[i] == 'd3d':
                d3d_count+=1
            else:
                east_count+=1
            
        return cmod_count, d3d_count, east_count

def generate_datasets(file_name: str, test_size: float, val_size: float, device, included_machines=['cmod','d3d','east'], new_machine='cmod', case=4, balance=False):
    data = PlasmaDataset.load_file(os.path.dirname(__file__)+'/../data/'+file_name)

    # Convert to list of dicts
    new_data = []
    old_data = []

    shot_data = []
    for shot in data.values():
        if shot['machine'] in included_machines:
            if shot['machine'] == new_machine:
                new_data.append(shot)
            else:
                old_data.append(shot)

    random.shuffle(new_data)
    random.shuffle(old_data)
    
    test_dataset = new_data[:int(len(new_data)*test_size)]
    new_data = new_data[int(len(new_data)*(test_size)):]

    if case == 1:
        train_dataset = old_data
    elif case == 2:
        train_dataset = new_data[:20]+old_data
    elif case == 3:
        train_dataset = new_data+old_data
    else:
        train_dataset = new_data


    if balance:
        train_dataset = balance_data(train_dataset)

    random.shuffle(train_dataset)

    val_dataset = train_dataset[-int(len(new_data)*(val_size)):]
    train_dataset = train_dataset[:-int(len(new_data)*(val_size))]

    # Get datasets

    train_dataset = PlasmaDataset(
        shot_data=train_dataset[:100],
        device=device
    )

    test_dataset = PlasmaDataset(
        shot_data=test_dataset,
        device=device
    )
    val_dataset = PlasmaDataset(
        shot_data=val_dataset,
        device=device
    )
    scaler = train_dataset.scale()
    test_dataset.scale_w_scaler(scaler)
    val_dataset.scale_w_scaler(scaler)

    train_dataset.move_to_device()
    test_dataset.move_to_device()
    val_dataset.move_to_device()

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
        [df["predict_inputs_embeds"].to(dtype=torch.float32) for df in dataset],
        padding_value=-100,
        batch_first=True)
    output['labels'] = torch.stack([df['label'].to(dtype=torch.float32) for df in dataset]).unsqueeze(-1)

    return output

def distort_dataset(dataset, model, d_reps, nd_reps):
    with torch.no_grad():
        new_predict_inputs_embeds = []
        new_inputs_embeds = []
        new_labels = []
        new_machines = []
        for i, data in enumerate(dataset.predict_inputs_embeds):
            if dataset.labels[i] == 1:
                for j in range(d_reps):
                    new_predict_inputs_embeds.append(model(data.unsqueeze(0)).squeeze())
                    new_inputs_embeds.append(model(dataset.inputs_embeds[i].unsqueeze(0), specified_distortion_budget=np.rand()).squeeze())
                    new_labels.append(dataset.labels[i])
                    new_machines.append(dataset.machines[i])
                    
            else:
                for j in range(nd_reps):
                    new_predict_inputs_embeds.append(model(data.unsqueeze(0)).squeeze())
                    new_inputs_embeds.append(model(dataset.inputs_embeds[i].unsqueeze(0)).squeeze())
                    new_labels.append(dataset.labels[i])
                    new_machines.append(dataset.machines[i])

        new_dataset = copy.deepcopy(dataset)
        new_dataset.predict_inputs_embeds = new_predict_inputs_embeds
        new_dataset.inputs_embeds = new_inputs_embeds
        new_dataset.labels = new_labels
        new_dataset.machines = new_machines

        return new_dataset
