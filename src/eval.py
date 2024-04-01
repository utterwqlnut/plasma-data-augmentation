from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

def compute_metrics(out, labels):
    binary_out = np.round(out.detach().numpy())
    
    return accuracy_score(labels, binary_out), f1_score(labels, binary_out), roc_auc_score(labels, binary_out)

def plot_view(model, input):
    input = input.to(torch.float32)

    fig, ax = plt.subplots(4,3)
    for i in range(12):
        idx1 = i%4
        idx2 = i//4
        ax[idx1][idx2].plot(np.transpose(input.squeeze().detach().numpy()[:,i]), label='Original')
        ax[idx1][idx2].plot(np.transpose(model(input).squeeze().detach().numpy()[:,i]), label='View')
        ax[idx1][idx2].legend()

    wandb.log({"Example View":wandb.Image(fig)})

def compute_metrics_after_training(model, test_dataset):
    out = []
    labels = []
    for sample in test_dataset:
        out.append(model(sample['predict_inputs_embeds'].unsqueeze(0).to(torch.float32)).squeeze())
        labels.append(sample['label'])
    
    accuracy, f1, auc = compute_metrics(torch.stack(out),torch.stack(labels))
    wandb.log({"Post Hoc Accuracy": accuracy, "Post Hoc F1": f1, "Post Hoc AUC": auc})