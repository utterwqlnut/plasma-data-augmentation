from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import torch

def compute_metrics(out, labels):
    binary_out = np.round(out.detach().numpy())
    
    return accuracy_score(labels, binary_out), f1_score(labels, binary_out), roc_auc_score(labels, binary_out)