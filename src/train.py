from data import PlasmaDataset, generate_datasets
import os
from torch.utils.data import DataLoader
import torch
import copy
from eval import compute_metrics
import math
import wandb
import numpy as np
from eval import plot_view

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

def train_post_hoc(train_dataloader, val_dataloader, optim, loss_fn, model, viewmaker, viewmaker_aug, varied_distortion_budget, max_distortion_budget, save_metric, num_epochs):
    if viewmaker_aug:
        prefix = 'Aug '
    else:
        prefix = 'No Aug '

    # Train PostHoc LSTM
    max_metric = 1e-10
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

            with torch.no_grad():
                if viewmaker_aug:
                    if not varied_distortion_budget:
                        inputs = viewmaker(inputs)
                    else:
                        inputs = viewmaker(inputs, np.random.rand()*max_distortion_budget)

            # Not training the viewmaker just post hoc
            #inputs = inputs.detach()

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


        if running_metric/len(val_dataloader)>max_metric:
            best_model = copy.deepcopy(model)
            max_metric = running_metric/len(val_dataloader)
            best_model.metric = max_metric
        
        wandb.log({prefix+'Post Hoc Epoch': epoch+1})
        wandb.log({prefix+'Post Hoc Training Loss': train_running_loss/len(train_dataloader),
                   prefix+'Post Hoc Validation Loss': val_running_loss/len(val_dataloader),
                   prefix+'Validation Accuracy': running_accuracy/len(val_dataloader),
                   prefix+'Validation F1': running_f1/len(val_dataloader),
                   prefix+'Validation AUC': running_auc/len(val_dataloader)})

        print(f"Epoch: {epoch+1}")
        print(f"Losses: Training Loss: {train_running_loss/len(train_dataloader)} Validation Loss: {val_running_loss/len(val_dataloader)}")
        print(f"Metrics: Validation Accuracy: {running_accuracy/len(val_dataloader)} Validation F1: {running_f1/len(val_dataloader)} Validation AUC: {running_auc/len(val_dataloader)}")

    return best_model

class ViewMakerTrainer():
    def __init__(self, train_dataset, val_dataset,batch_size, t, v_loss_weight, collate_fn, viewmaker, encoder, v_lr, e_lr):
        self.configure_dataloaders(train_dataset, val_dataset, batch_size, collate_fn)
        self.t = t
        self.v_loss_weight = v_loss_weight
        self.v_loss_weight
        self.viewmaker = viewmaker
        self.encoder = encoder
        self.configure_optimizers(v_lr, e_lr)

    def configure_optimizers(self, v_lr, e_lr):
        self.encoder_optim = torch.optim.AdamW(list(self.encoder.parameters()), lr=e_lr)
        self.viewmaker_optim = torch.optim.AdamW(self.viewmaker.parameters(), lr=v_lr)


    def configure_dataloaders(self, train_dataset, val_dataset, batch_size, collate_fn):
        self.train_dataloader = DataLoader(train_dataset, batch_size=2*batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_dataloader = DataLoader(val_dataset, batch_size=2*batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, num_epochs):
        torch.autograd.set_detect_anomaly(True)
        val_step = 0

        for epoch in range(num_epochs): 
            train_running_e_loss = 0
            train_running_v_loss = 0
            val_running_e_loss = 0
            val_running_v_loss = 0

            for i, data in enumerate(self.train_dataloader):

                #if data['inputs_embeds'].shape[0] % 2 !=0:
                #    data['inputs_embeds'] = data['inputs_embeds'][:-1]

                #x1, x2 = torch.chunk(data['inputs_embeds'],2)
                x1 = data['inputs_embeds']
                x2 = data['inputs_embeds'].clone()
                view1 = self.viewmaker(x1)
                view2 = self.viewmaker(x2)

                encoder_loss, _ = AdversarialSimCLRLoss(self.encoder(view1.detach()), self.encoder(view2.detach()), self.t, self.v_loss_weight).get_loss()

                self.encoder.zero_grad()
                encoder_loss.backward()
                self.encoder_optim.step()

                _, viewmaker_loss = AdversarialSimCLRLoss(self.encoder(view1), self.encoder(view2), self.t, self.v_loss_weight).get_loss()

                self.viewmaker.zero_grad()
                viewmaker_loss.backward()
                self.viewmaker_optim.step()

                train_running_e_loss += encoder_loss.item()
                train_running_v_loss += viewmaker_loss.item()

            for i, data in enumerate(self.val_dataloader):
                
                #if data['inputs_embeds'].shape[0] % 2 !=0:
                #    data['inputs_embeds'] = data['inputs_embeds'][:-1]
                self.viewmaker.eval()

                x1 = data['inputs_embeds']
                x2 = data['inputs_embeds'].clone()
                
                view1_embd = self.encoder(self.viewmaker(x1))
                view2_embd = self.encoder(self.viewmaker(x2))

                encoder_loss, viewmaker_loss = AdversarialSimCLRLoss(view1_embd, view2_embd, self.t, self.v_loss_weight).get_loss()

                val_running_e_loss += encoder_loss.item()
                val_running_v_loss += viewmaker_loss.item()

                if val_step%10 == 0:
                    x = data['inputs_embeds'][0][:,0]
                    length = len(x[x!=-100])
                    plot_view(self.viewmaker, data['inputs_embeds'][0].unsqueeze(0)[:,:length], title='Example Val View')
                
                val_step+=1

                self.viewmaker.train()

            wandb.log({'Epoch': epoch+1})
            wandb.log({'Encoder Training Loss': train_running_e_loss/len(self.train_dataloader),
                   'Viewmaker Training Loss': train_running_v_loss/len(self.train_dataloader),
                   'Viewmaker Validation Loss': val_running_v_loss/len(self.val_dataloader),
                   'Encoder Validation Loss': val_running_e_loss/len(self.val_dataloader)})

            print(f"Epoch: {epoch+1}")
            print(f"Train Losses: Training Encoder Loss: {train_running_e_loss/len(self.train_dataloader)} Training Viewmaker Loss: {train_running_v_loss/len(self.train_dataloader)}")
            print(f"Val Losses: Val Encoder Loss: {val_running_e_loss/len(self.val_dataloader)} Val Viewmaker Loss: {val_running_v_loss/len(self.val_dataloader)}")


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))

class SimCLRObjective(torch.nn.Module):

    def __init__(self, outputs1, outputs2, t, push_only=False):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.t = t
        self.push_only = push_only

    def get_loss(self):
        batch_size = self.outputs1.size(0)  # batch_size x out_dim
        witness_score = torch.sum(self.outputs1 * self.outputs2, dim=1)
        if self.push_only:
            # Don't pull views together.
            witness_score = 0
        outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        witness_norm = self.outputs1 @ outputs12.T
        witness_norm = torch.logsumexp(witness_norm / self.t, dim=1) - math.log(2 * batch_size)
        loss = -torch.mean(witness_score / self.t - witness_norm)
        return loss

class AdversarialSimCLRLoss(object):

    def __init__(
        self,
        embs1,
        embs2,
        t=0.07,
        view_maker_loss_weight=1.0,
        **kwargs
    ):
        '''Adversarial version of SimCLR loss.
        
        Args:
            embs1: embeddings of the first views of the inputs
            embs1: embeddings of the second views of the inputs
            t: temperature
            view_maker_loss_weight: how much to weight the view_maker loss vs the encoder loss
        '''
        self.embs1 = embs1
        self.embs2 = embs2
        self.t = t
        self.view_maker_loss_weight = view_maker_loss_weight

        self.normalize_embeddings()

    def normalize_embeddings(self):
        self.embs1 = l2_normalize(self.embs1)
        self.embs2 = l2_normalize(self.embs2)

    def get_loss(self):
        '''Return scalar encoder and view-maker losses for the batch'''
        simclr_loss = SimCLRObjective(self.embs1, self.embs2, self.t)
        encoder_loss = simclr_loss.get_loss()
        view_maker_loss = -encoder_loss * self.view_maker_loss_weight
        return encoder_loss, view_maker_loss



