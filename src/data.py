import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import sys
import os
import pickle

class PlasmaDataset(Dataset):

    def __init__(self, shot_data, cutoff_steps=4):
        self.cutoff_steps = cutoff_steps
        self.labels = []
        self.machines = []
        self.inputs_embeds = []
        self.predict_inputs_embeds = []

        for i in range(len(shot_data)):
            item = shot_data[i]
            if len(item['data'])<=cutoff_steps:
                continue

            self.labels.append(torch.tensor(item['label']))
            self.machines.append(item['machine'])
            self.inputs_embeds.append(torch.tensor(item['data'].values))
            self.predict_inputs_embeds.append(torch.tensor(item['data'].values[:-cutoff_steps]))

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

def generate_datasets(file_name: str, train_size: float, test_size: float, val_size: float, included_machines=['cmod','d3d','east']):
    data = PlasmaDataset.load_file(os.path.dirname(__file__)+'/../data/'+file_name)
    
    # Convert to list of dicts
    shot_data = []
    for shot in data.values():
        if shot['machine'] in included_machines:
            shot_data.append(shot)
    
    # Get datasets
    train_dataset = PlasmaDataset(
        shot_data=shot_data[:int(train_size*len(shot_data))]
    )
    test_dataset = PlasmaDataset(
        shot_data=shot_data[len(train_dataset):len(train_dataset)+int(test_size*len(shot_data))]
    )
    val_dataset = PlasmaDataset(
        shot_data=shot_data[len(test_dataset):]
    )

    return train_dataset, test_dataset, val_dataset

def collate_fn(dataset):
    output = {}

    output['inputs_embeds'] = pad_sequence(
        [df["predict_inputs_embeds"].to(dtype=torch.float32) for df in dataset],
        padding_value=-100,
        batch_first=True)
    output['labels'] = torch.stack([df['label'].to(dtype=torch.float32) for df in dataset]).unsqueeze(-1)

    return output
      