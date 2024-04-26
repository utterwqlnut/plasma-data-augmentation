from data import PlasmaDataset, generate_datasets, BatchSampler
import os
from torch.utils.data import DataLoader
import torch
import copy
from eval import compute_metrics, compute_metrics_during_training, plot_view
import math
import wandb
import numpy as np
from accelerate import Accelerator

accelerator = Accelerator()

torch.manual_seed(42)

def train_post_hoc(train_dataloader, val_dataloader, val_dataset, optim, loss_fn, model, viewmaker, viewmaker_aug, save_metric, num_epochs):
    
    model, train_dataloader, val_dataloader, optim = accelerator.prepare(model, train_dataloader, val_dataloader, optim)
    
    total_steps = 0
    eval_steps = 2
    train_running_loss = 0

    if viewmaker_aug:
        prefix = 'Aug '
    else:
        prefix = 'No Aug '

    # Train PostHoc LSTM
    max_metric = 1e-10
    best_model = copy.deepcopy(model)

    for epoch in range(num_epochs):
        #train_running_loss = 0

        for i, data in enumerate(train_dataloader):

            inputs, labels = (data['inputs_embeds'], data['labels'])

            optim.zero_grad()
            
            out = model(inputs)

            loss = loss_fn(out, labels) 

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optim.step()
            train_running_loss += loss.item()

            total_steps+=1

            if total_steps % eval_steps == 0:
                val_running_loss = 0

                model.eval()

                for i, data in enumerate(val_dataloader):
                    inputs, labels = (data['inputs_embeds'], data['labels'])

                    out=model(inputs)
                    loss = loss_fn(out,labels)
                    
                    val_running_loss+=loss.item()

                accuracy, f1, auc = compute_metrics_during_training(model, val_dataset)
                if save_metric == 'accuracy':
                    metric_result = accuracy
                elif save_metric == 'f1':
                    metric_result = f1
                elif save_metric == 'auc':
                    metric_result = auc
                else:
                    print('Selected save metric does not exist defaulting to accuracy')
                    metric_result = accuracy

                model.train()

                if metric_result>max_metric:
                    best_model = copy.deepcopy(model)
                    max_metric = metric_result
                    best_model.metric = max_metric

                wandb.log({prefix+'Post Hoc Epoch': total_steps/len(train_dataloader)})
                wandb.log({prefix+'Post Hoc Training Loss': train_running_loss/eval_steps,
                        prefix+'Post Hoc Validation Loss': val_running_loss/len(val_dataloader),
                        prefix+'Validation Accuracy': accuracy,
                        prefix+'Validation F1': f1,
                        prefix+'Validation AUC': auc})

                print(f"Epoch: {total_steps/len(train_dataloader)}")
                print(f"Losses: Training Loss: {train_running_loss/eval_steps} Validation Loss: {val_running_loss/len(val_dataloader)}")
                print(f"Metrics: Validation Accuracy: {accuracy} Validation F1: {f1} Validation AUC: {auc}")

                train_running_loss = 0


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
        train_lengths = []
        val_lengths = []

        for data in train_dataset:
            train_lengths.append(len(data['inputs_embeds']))
        
        for data in val_dataset:
            val_lengths.append(len(data['inputs_embeds']))

        self.train_dataloader = DataLoader(train_dataset, batch_sampler=BatchSampler(train_lengths,batch_size), collate_fn=collate_fn)
        self.val_dataloader = DataLoader(val_dataset, batch_sampler=BatchSampler(val_lengths,batch_size), collate_fn=collate_fn)

    def train(self, num_epochs):
        # Accelerator setup
        self.viewmaker, self.encoder, self.encoder_optim, self.viewmaker_optim, self.train_dataloader, self.val_dataloader = accelerator.prepare(
            self.viewmaker, self.encoder, self.encoder_optim, self.viewmaker_optim, self.train_dataloader, self.val_dataloader)
        
        val_count = 0
        eval_steps = 2
        total_steps = 0
        train_running_e_loss = 0
        train_running_v_loss = 0

        for epoch in range(num_epochs): 
            #train_running_e_loss = 0
            #train_running_v_loss = 0
           
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
                accelerator.backward(encoder_loss)
                self.encoder_optim.step()

                _, viewmaker_loss = AdversarialSimCLRLoss(self.encoder(view1), self.encoder(view2), self.t, self.v_loss_weight).get_loss()

                self.viewmaker.zero_grad()
                accelerator.backward(viewmaker_loss)
                self.viewmaker_optim.step()

                train_running_e_loss += encoder_loss.item()
                train_running_v_loss += viewmaker_loss.item()

                total_steps+=1

                if total_steps % eval_steps == 0:
                    self.viewmaker.eval()
                    val_running_e_loss = 0
                    val_running_v_loss = 0
                
                    for i, data in enumerate(self.val_dataloader):
                        
                        #if data['inputs_embeds'].shape[0] % 2 !=0:
                        #    data['inputs_embeds'] = data['inputs_embeds'][:-1]

                        x1 = data['inputs_embeds']
                        x2 = data['inputs_embeds'].clone()
                        
                        view1_embd = self.encoder(self.viewmaker(x1))
                        view2_embd = self.encoder(self.viewmaker(x2))

                        encoder_loss, viewmaker_loss = AdversarialSimCLRLoss(view1_embd, view2_embd, self.t, self.v_loss_weight).get_loss()

                        val_running_e_loss += encoder_loss.item()
                        val_running_v_loss += viewmaker_loss.item()

                        if val_count%10 == 0 and i==0:
                            x = data['inputs_embeds'][0][:,0]
                            length = len(x[x!=-100])
                            plot_view(self.viewmaker, data['inputs_embeds'][0].unsqueeze(0)[:,:length], title='Example Val View')
                        
                    val_count+=1
        
                    self.viewmaker.train()

                    self.viewmaker.flag = 'something'
                    wandb.log({'Epoch': total_steps/len(self.train_dataloader)})
                    wandb.log({'Encoder Training Loss': train_running_e_loss/eval_steps,
                        'Viewmaker Training Loss': train_running_v_loss/eval_steps,
                        'Viewmaker Validation Loss': val_running_v_loss/len(self.val_dataloader),
                        'Encoder Validation Loss': val_running_e_loss/len(self.val_dataloader)})

                    print(f"Epoch: {total_steps/len(self.train_dataloader)}")
                    print(f"Train Losses: Training Encoder Loss: {train_running_e_loss/(eval_steps)} Training Viewmaker Loss: {train_running_v_loss/(eval_steps)}")
                    print(f"Val Losses: Val Encoder Loss: {val_running_e_loss/len(self.val_dataloader)} Val Viewmaker Loss: {val_running_v_loss/len(self.val_dataloader)}")

                    train_running_e_loss = 0
                    train_running_v_loss = 0


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



