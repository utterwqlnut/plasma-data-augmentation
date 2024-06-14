from data import PlasmaDataset, generate_datasets, BatchSampler
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from models import series_decomp
import torch
import copy
from eval import compute_metrics, compute_metrics_during_training, plot_view
import math
import wandb
import numpy as np

torch.manual_seed(42)

def train_post_hoc(train_dataloader, val_dataloader, val_dataset, optim, loss_fn, model, viewmaker, viewmaker_aug, save_metric, num_epochs, fabric):

    model, optim = fabric.setup(model,optim)
    scheduler = ExponentialLR(optim, gamma=0.9)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    total_steps = 0
    eval_steps = 100 # Low steps for testing change on cluster to approx 100
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

            fabric.backward(loss)

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

                accuracy, f1, auc = compute_metrics_during_training(model, val_dataset, fabric.device)
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

                wandb.log({prefix+'Post Hoc Epoch': total_steps/len(train_dataloader), prefix+'Post Hoc Total Steps': total_steps})
                wandb.log({prefix+'Post Hoc Learning Rate': scheduler.get_last_lr()[0]})
                wandb.log({prefix+'Post Hoc Training Loss': train_running_loss/eval_steps,
                        prefix+'Post Hoc Validation Loss': val_running_loss/len(val_dataloader),
                        prefix+'Validation Accuracy': accuracy,
                        prefix+'Validation F1': f1,
                        prefix+'Validation AUC': auc})

                print(f"Epoch: {total_steps/len(train_dataloader)}")
                print(f"Losses: Training Loss: {train_running_loss/eval_steps} Validation Loss: {val_running_loss/len(val_dataloader)}")
                print(f"Metrics: Validation Accuracy: {accuracy} Validation F1: {f1} Validation AUC: {auc}")

                train_running_loss = 0

                scheduler.step()

    return best_model

class ViewMakerTrainer():
    def __init__(self, train_dataset, val_dataset,batch_size, t, v_loss_weight, collate_fn, viewmaker, encoder, mlp, v_lr, e_lr, m_lr, fabric):
        self.fabric=fabric
        self.configure_dataloaders(train_dataset, val_dataset, batch_size, collate_fn)
        self.t = t
        self.loss = SimCLR_Loss(batch_size,t)
        self.v_loss_weight = v_loss_weight
        self.v_loss_weight
        self.viewmaker = viewmaker
        self.encoder = encoder
        self.configure_optimizers(v_lr, e_lr, m_lr)

        self.viewmaker, self.viewmaker_optim = self.fabric.setup(self.viewmaker, self.viewmaker_optim)
        self.encoder, self.encoder_optim = self.fabric.setup(self.encoder, self.encoder_optim)

        self.train_dataloader, self.val_dataloader = fabric.setup_dataloaders(self.train_dataloader, self.val_dataloader)

    def configure_optimizers(self, v_lr, e_lr, m_lr):
        self.encoder_optim = torch.optim.AdamW(self.encoder.parameters(), lr=e_lr)
        self.viewmaker_optim = torch.optim.AdamW(self.viewmaker.parameters(), lr=v_lr)
        self.scheduler_e = ExponentialLR(self.encoder_optim, gamma=0.9)
        self.scheduler_v = ExponentialLR(self.viewmaker_optim, gamma=0.9)


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
        val_count = 0
        eval_steps = 100 # Low steps for testing change on cluster to approx 100
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
                x2 = x1.clone()
                x3 = x2.clone()
                labels = data['labels']

                view1 = self.viewmaker(x1)
                view2 = self.viewmaker(x2)

                encoder_loss = self.loss(self.encoder(view1.detach()), self.encoder(view2.detach()))

                self.encoder.zero_grad()
                self.fabric.backward(encoder_loss)
                self.encoder_optim.step()

                viewmaker_loss = -self.v_loss_weight*self.loss(self.encoder(view1), self.encoder(view2))

                self.viewmaker.zero_grad()
                self.fabric.backward(viewmaker_loss)
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

                        encoder_loss = self.loss(view1_embd, view2_embd)
                        viewmaker_loss = -self.v_loss_weight*encoder_loss

                        val_running_e_loss += encoder_loss.item()
                        val_running_v_loss += viewmaker_loss.item()

                        if val_count%10 == 0 and i==0:
                            x = data['inputs_embeds'][0][:,0]
                            length = len(x[x!=-100])
                            plot_view(self.viewmaker, data['inputs_embeds'][0].unsqueeze(0)[:,:length], title='Example Val View')

                    val_count+=1

                    self.viewmaker.train()

                    self.viewmaker.flag = 'something'
                    wandb.log({'Epoch': total_steps/len(self.train_dataloader), 'Total Steps': total_steps})
                    wandb.log({'Viewmaker Learning Rate': self.scheduler_v.get_last_lr()[0]})
                    wandb.log({'Encoder Learning Rate': self.scheduler_e.get_last_lr()[0]})
                    wandb.log({'Encoder Training Loss': train_running_e_loss/eval_steps,
                        'Viewmaker Training Loss': train_running_v_loss/eval_steps,
                        'Viewmaker Validation Loss': val_running_v_loss/len(self.val_dataloader),
                        'Encoder Validation Loss': val_running_e_loss/len(self.val_dataloader)})

                    print(f"Epoch: {total_steps/len(self.train_dataloader)}")
                    print(f"Train Losses: Training Encoder Loss: {train_running_e_loss/(eval_steps)} Training Viewmaker Loss: {train_running_v_loss/(eval_steps)}")
                    print(f"Val Losses: Val Encoder Loss: {val_running_e_loss/len(self.val_dataloader)} Val Viewmaker Loss: {val_running_v_loss/len(self.val_dataloader)}")

                    train_running_e_loss = 0
                    train_running_v_loss = 0

            self.scheduler_e.step()
            self.scheduler_v.step()

class ViewMakerPreTrainer():
    def __init__(self, train_dataset, val_dataset,batch_size, collate_fn, viewmaker, v_lr, fabric):
        self.fabric=fabric
        self.configure_dataloaders(train_dataset, val_dataset, batch_size, collate_fn)

        self.viewmaker = viewmaker
        self.loss_fn = torch.nn.MSELoss()

        self.configure_optimizers(v_lr)

        self.viewmaker, self.viewmaker_optim = self.fabric.setup(self.viewmaker, self.viewmaker_optim)

        self.train_dataloader, self.val_dataloader = fabric.setup_dataloaders(self.train_dataloader, self.val_dataloader)

    def configure_optimizers(self, v_lr):
        self.viewmaker_optim = torch.optim.AdamW(self.viewmaker.parameters(), lr=v_lr)
        self.scheduler_v = ExponentialLR(self.viewmaker_optim, gamma=0.9)


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
        val_count = 0
        eval_steps = 2 # Low steps for testing change on cluster to approx 100
        total_steps = 0
        train_running_v_loss = 0

        for epoch in range(num_epochs):
            #train_running_e_loss = 0
            #train_running_v_loss = 0

            for i, data in enumerate(self.train_dataloader):

                #if data['inputs_embeds'].shape[0] % 2 !=0:
                #    data['inputs_embeds'] = data['inputs_embeds'][:-1]

                #x1, x2 = torch.chunk(data['inputs_embeds'],2)

                x1 = data['inputs_embeds']
                x2 = x1.clone()

                view1 = self.viewmaker(x1)
                view2 = self.viewmaker(x2)

                viewmaker_loss = -1*self.loss_fn(view1,view2)
                self.viewmaker.zero_grad()
                self.fabric.backward(viewmaker_loss)
                self.viewmaker_optim.step()

                train_running_v_loss += viewmaker_loss.item()

                total_steps+=1

                if total_steps % eval_steps == 0:
                    self.viewmaker.eval()
                    val_running_v_loss = 0

                    for i, data in enumerate(self.val_dataloader):

                        #if data['inputs_embeds'].shape[0] % 2 !=0:
                        #    data['inputs_embeds'] = data['inputs_embeds'][:-1]
                        viewmaker_loss = -1*self.loss_fn(view1,view2)
                        x1 = data['inputs_embeds']
                        x2 = data['inputs_embeds'].clone()

                        val_running_v_loss += viewmaker_loss.item()

                        if val_count%10 == 0 and i==0:
                            x = data['inputs_embeds'][0][:,0]
                            length = len(x[x!=-100])
                            plot_view(self.viewmaker, data['inputs_embeds'][0].unsqueeze(0)[:,:length], title='Example Val Pretrain View')

                    val_count+=1

                    self.viewmaker.train()

                    wandb.log({'Pretraining Epoch': total_steps/len(self.train_dataloader), 'Pretraining Total Steps': total_steps})
                    wandb.log({'Pretraining Viewmaker Learning Rate': self.scheduler_v.get_last_lr()[0]})
                    wandb.log({'Viewmaker Pretraining Loss': train_running_v_loss/eval_steps,
                        'Viewmaker Pretraining Validation Loss': val_running_v_loss/len(self.val_dataloader),})

                    print(f"Pretraining Epoch: {total_steps/len(self.train_dataloader)}")
                    print(f"Pretraining Viewmaker Loss: {train_running_v_loss/(eval_steps)}")
                    print(f"Val Pretraining Viewmaker Loss: {val_running_v_loss/len(self.val_dataloader)}")

                    train_running_v_loss = 0

                    self.scheduler_v.step()

def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))

class SimCLR_Loss(torch.nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = torch.nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

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