from data import PlasmaDataset, generate_datasets, collate_fn
from models import PlasmaLSTM
import os
from torch.utils.data import DataLoader
import torch
import copy
from eval import compute_metrics
from train import train_post_hoc

# Get Dataset
file_name = 'Full_HDL_dataset_unnormalized_no_nan_column_names_w_shot.pickle'
train_dataset, test_dataset, val_dataset = generate_datasets(file_name,0.8,0.1,0.1,included_machines=['east'])

# Train Viewmaker
# Generate Views
# Add to train dataset (Need to add method for that in PlasmaDataset)

# Train an Post Hoc LSTM
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True,collate_fn=collate_fn)


model = PlasmaLSTM(12,5,12)
adam = torch.optim.Adam(params=model.parameters(),lr=1e-3)
loss_fn = torch.nn.BCELoss()
save_metric = 'accuracy'

best_model = train_post_hoc(train_dataloader=train_dataloader, val_dataloader=test_dataloader, model=model,optim=adam,loss_fn=loss_fn, save_metric=save_metric, num_epochs=10)

# Evaluate on test dataset