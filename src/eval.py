from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

torch.manual_seed(42)

def compute_metrics(out, labels):
    labels = labels.cpu().detach().numpy()
    binary_out = np.round(out.cpu().detach().numpy())

    return accuracy_score(labels, binary_out), f1_score(labels, binary_out), roc_auc_score(labels, binary_out)

def plot_view(model, input, title='Example Test View'):
    input = input.to(torch.float32)

    fig, ax = plt.subplots(4,3)
    for i in range(12):
        idx1 = i%4
        idx2 = i//4
        ax[idx1][idx2].plot(np.transpose(input.squeeze().cpu().detach().numpy()[:,i]), label='Original')
        ax[idx1][idx2].plot(np.transpose(model(input).squeeze().cpu().detach().numpy()[:,i]), label='View')
        ax[idx1][idx2].legend()

    wandb.log({title:wandb.Image(fig)})

def compute_metrics_after_training(model, test_dataset, prefix):
    model.eval()

    out = []
    labels = []
    for sample in test_dataset:
        out.append(model(sample['inputs_embeds'][:-test_dataset.cutoff_steps].unsqueeze(0).to(torch.float32)).squeeze())
        labels.append(sample['label'])
    
    accuracy, f1, auc = compute_metrics(torch.stack(out),torch.stack(labels))
    wandb.log({prefix+"Post Hoc Accuracy": accuracy, prefix+"Post Hoc F1": f1, prefix+"Post Hoc AUC": auc})

def compute_metrics_during_training(model, val_dataset):
    out = []
    labels = []
    for sample in val_dataset:
        out.append(model(sample['inputs_embeds'][:-val_dataset.cutoff_steps].unsqueeze(0).to(torch.float32)).squeeze())
        labels.append(sample['label'])
    
    accuracy, f1, auc = compute_metrics(torch.stack(out),torch.stack(labels)) 
    return accuracy, f1, auc