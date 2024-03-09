from data import PlasmaDataset, generate_datasets, collate_fn
from models import PlasmaLSTM
import os
from torch.utils.data import DataLoader
import torch
import copy
from eval import compute_metrics


def train_post_hoc(train_dataloader, val_dataloader, optim, loss_fn, model, save_metric, num_epochs):
    # Train PostHoc LSTM
    min_metric = 1e10
    best_model = copy.deepcopy(model)

    for epoch in range(num_epochs):
        train_running_loss = 0
        val_running_loss = 0
        running_metric = 0
        running_accuracy = 0
        running_f1 = 0
        running_auc = 0

        for i, data in enumerate(train_dataloader):

            inputs, labels = (data['inputs_embeds'], data['labels'])
            optim.zero_grad()
            
            out = model(inputs)

            loss = loss_fn(out, labels) 
            loss.backward()

            optim.step()
            train_running_loss += loss.item()

        for i, data in enumerate(val_dataloader):
            inputs, labels = (data['inputs_embeds'], data['labels'])

            out=model(inputs)
            loss = loss_fn(out,labels)

            accuracy, f1, auc = compute_metrics(out,labels)
            running_accuracy += accuracy
            running_f1 += f1
            running_auc += auc

            if save_metric == 'accuracy':
                metric_result = accuracy
            elif save_metric == 'f1':
                metric_result = f1
            elif save_metric == 'auc':
                metric_result = auc
            else:
                print('Selected save metric does not exist defaulting to accuracy')
                metric_result = accuracy

            running_metric += metric_result
            val_running_loss+=loss.item()

        if running_metric<min_metric:
            best_model = copy.deepcopy(model)
            min_metric = running_metric

        print(f"Epoch: {epoch+1}")
        print(f"Losses: Training Loss: {train_running_loss/len(train_dataloader)} Validation Loss: {train_running_loss/len(train_dataloader)}")
        print(f"Metrics: Validation Accuracy: {running_accuracy/len(val_dataloader)} Validation F1: {running_f1/len(val_dataloader)} Validation AUC: {running_auc/len(val_dataloader)}")

    return best_model