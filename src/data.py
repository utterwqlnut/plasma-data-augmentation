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


class PlasmaDataset(Dataset):

    def __init__(self, shot_data, cutoff_steps=8):
        self.cutoff_steps = cutoff_steps
        self.labels = []
        self.machines = []
        self.inputs_embeds = []
        self.max_length = 2048

        for i in range(len(shot_data)):
            item = shot_data[i]
            if len(item['data'])<=25 or len(item['data'])>=self.max_length:
                continue
            self.labels.append(torch.tensor(item['label']))
            self.machines.append(item['machine'])
            self.inputs_embeds.append(torch.tensor(item['data'].values,dtype=torch.float32))


    def __getitem__(self, index):
        return {'label':self.labels[index],
                'machine': self.machines[index],
                'inputs_embeds': self.inputs_embeds[index],
                'cutoff_steps': self.cutoff_steps}

    def __len__(self):
        return len(self.labels)

    def load_file(file_name):
        file = open(file_name,'rb')
        data = pickle.load(file)

        return data

    def scale(self):
        combined_full = torch.cat(self.inputs_embeds)

        scaler = StandardScaler().fit(combined_full)

        with torch.no_grad():
            for i in range(len(self.inputs_embeds)):
                self.inputs_embeds[i] = torch.tensor(normalize(scaler.transform(self.inputs_embeds[i])),dtype=torch.float32)

        return scaler

    def scale_w_scaler(self, scaler):
        with torch.no_grad():
            for i in range(len(self.inputs_embeds)):
                self.inputs_embeds[i] = torch.tensor(normalize(scaler.transform(self.inputs_embeds[i])),dtype=torch.float32)

    def move_to_device(self):
        for i in range(len(self.inputs_embeds)):
            self.inputs_embeds[i]=self.inputs_embeds[i].to(self.device)
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

    def window(self, smooth_labels, num_ecs):

        new_data = []
        new_labels = []
        new_machines = []
        end_cutoffs = [8,6,4,2,1][:num_ecs]
        smoothed_disruptivities = [0.8,0.85,0.9,0.95,1]
        end_time_index = 0

        for i,label in enumerate(self.labels):
            if label == 1:
                for j, ec in enumerate(end_cutoffs):
                    if smooth_labels:
                        new_labels.append(torch.tensor(smoothed_disruptivities[j]))
                    else:
                        new_labels.append(torch.tensor(1))
                    new_machines.append(self.machines[i])
                    new_data.append(self.inputs_embeds[i][:-ec])
            else:
                nd_random_ends = random.sample(range(10,len(self.inputs_embeds[i])),2)
                for end in nd_random_ends:
                    new_labels.append(torch.tensor(0))
                    new_machines.append(self.machines[i])
                    new_data.append(self.inputs_embeds[i][:end+1])

        self.inputs_embeds = new_data
        self.machines = new_machines
        self.labels = new_labels
def generate_datasets(file_name: str, test_size: float, val_size: float, included_machines=['cmod','d3d','east'], new_machine='cmod', case=4, balance=False):
    random.seed(42)
    data = PlasmaDataset.load_file(os.path.dirname(__file__)+'/../data/'+file_name)

    # Convert to list of dicts
    new_data_d = []
    new_data_nd = []
    old_data = []

    for shot in data.values():
        if shot['machine'] in included_machines:
            if shot['machine'] == new_machine:
                if shot['label']:
                    new_data_d.append(shot)
                else:
                    new_data_nd.append(shot)
            else:
                old_data.append(shot)

    random.shuffle(new_data_d)
    random.shuffle(new_data_nd)
    random.shuffle(old_data)

    test_dataset = new_data_d[:int(len(new_data_d)*test_size)]+new_data_nd[:int(len(new_data_nd)*test_size)]
    new_data_d = new_data_d[int(len(new_data_d)*(test_size)):]
    new_data_nd = new_data_nd[int(len(new_data_nd)*(test_size)):]

    if case == 1:
        train_dataset = old_data
    elif case == 2:
        train_dataset = new_data_nd+new_data_d[:20]+old_data
    elif case == 3:
        train_dataset = new_data_d+new_data_nd+old_data
    else:
        train_dataset = new_data_d+new_data_nd

    random.shuffle(train_dataset)

    val_dataset = train_dataset[-int(len(train_dataset)*(val_size)):]
    train_dataset = train_dataset[:-int(len(train_dataset)*(val_size))]


    # Get datasets

    train_dataset = PlasmaDataset(
        shot_data=train_dataset,    # keep during testing
    )
    test_dataset = PlasmaDataset(
        shot_data=test_dataset,     # keep during testing
    )
    val_dataset = PlasmaDataset(
        shot_data=val_dataset,      # keep during testing
    )
    scaler = train_dataset.scale()
    test_dataset.scale_w_scaler(scaler)
    val_dataset.scale_w_scaler(scaler)

    if balance:
        train_dataset = balance_machines(train_dataset)
        #train_dataset = balance_data(train_dataset)

    #train_dataset.move_to_device()
    #test_dataset.move_to_device()
    #val_dataset.move_to_device()

    return train_dataset, test_dataset, val_dataset

def balance_data(shot_dataset):
    new_data = []
    new_labels = []
    new_machines = []
    nd_count = 0

    num_d = 0
    for label in shot_dataset.labels:
        if label == 1:
            num_d+=1

    for i, label in enumerate(shot_dataset.labels):
        if label == 1:
            new_data.append(shot_dataset.inputs_embeds[i])
            new_labels.append(label)
            new_machines.append(shot_dataset.machines[i])
        elif nd_count<num_d:
            new_data.append(shot_dataset.inputs_embeds[i])
            new_labels.append(label)
            new_machines.append(shot_dataset.machines[i])
            nd_count+=1
        else:
            continue

    new_dataset = copy.deepcopy(shot_dataset)
    new_dataset.inputs_embeds = new_data
    new_dataset.labels = new_labels
    new_dataset.machines = new_machines

    return new_dataset

def balance_machines(shot_dataset):
    new_data = []
    new_labels = []
    new_machines = []

    num_cmod = 0
    num_d3d = 0
    num_east = 0
    for machine in shot_dataset.machines:
        if machine == "cmod":
            num_cmod+=1
        elif machine == "east":
            num_east+=1
        else:
            num_d3d+=1

    min_count = min(num_cmod,min(num_east,num_d3d))
    if min_count==0:
        min_count=min(num_east,num_d3d)
    num_cmod = 0
    num_d3d=0
    num_east=0
    for i, machine in enumerate(shot_dataset.machines):
        if machine == "cmod":
            new_data.append(shot_dataset.inputs_embeds[i])
            new_labels.append(shot_dataset.labels[i])
            new_machines.append(shot_dataset.machines[i])
        elif machine == "d3d" and num_d3d<min_count:
            new_data.append(shot_dataset.inputs_embeds[i])
            new_labels.append(shot_dataset.labels[i])
            new_machines.append(shot_dataset.machines[i])
            num_d3d+=1
        elif machine == "east" and num_east<min_count:
            new_data.append(shot_dataset.inputs_embeds[i])
            new_labels.append(shot_dataset.labels[i])
            new_machines.append(shot_dataset.machines[i])
            num_east+=1
        else:
            continue

    new_dataset = copy.deepcopy(shot_dataset)
    new_dataset.inputs_embeds = new_data
    new_dataset.labels = new_labels
    new_dataset.machines = new_machines

    return new_dataset

def post_hoc_collate_fn(dataset):
    output = {}

    output['inputs_embeds'] = pad_sequence(
        [df["inputs_embeds"].to(dtype=torch.float32) for df in dataset],
        padding_value=-100,
        batch_first=True)

    output['labels'] = torch.stack([df['label'].to(dtype=torch.float32) for df in dataset]).unsqueeze(-1)

    return output

def viewmaker_collate_fn(dataset):
    cutoffs = [2, 4, 6, 8, 10, 12]
    output = {}
    output['inputs_embeds'] = pad_sequence(
        [df["inputs_embeds"][:-cutoffs[int(random.random()*len(cutoffs))]].to(dtype=torch.float32) for df in dataset],
        padding_value=-100,
        batch_first=True)
    output['labels'] = torch.stack([df['label'].to(dtype=torch.float32) for df in dataset]).unsqueeze(-1)

    return output

class BatchSampler:
    def __init__(self, lengths, batch_size):
        self.lengths = lengths
        self.batch_size = batch_size

    def __iter__(self):
        size = len(self.lengths)
        indices = list(range(size))
        random.shuffle(indices)

        step = 100 * self.batch_size
        for i in range(0, size, step):
            pool = indices[i:i+step]
            pool = sorted(pool, key=lambda x: self.lengths[x])
            start_batches = list(range(0,len(pool),self.batch_size))[:-1]
            random.shuffle(start_batches)

            for j in range(len(start_batches)):
                yield pool[start_batches[j]:start_batches[j]+self.batch_size]

    def __len__(self):
        return len(self.lengths) // self.batch_size


def distort_dataset(dataset, model, d_reps, nd_reps, device):
    new_inputs_embeds = []
    with torch.no_grad():
        for i, data in enumerate(dataset.inputs_embeds):
            if dataset.labels[i] == 1:
                for j in range(d_reps):
                    new_inputs_embeds.append(model(dataset.inputs_embeds[i].unsqueeze(0).to(device)).squeeze())

            else:
                for j in range(nd_reps):
                    new_inputs_embeds.append(model(dataset.inputs_embeds[i].unsqueeze(0).to(device)).squeeze())

    dataset.inputs_embeds = new_inputs_embeds