import random
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, data_path):
        all_data = json.load(open(data_path, 'r'))
        self.human_score_ls = []
        self.logits_ls = []
        self.weighted_score_ls = []

        for data in all_data:
            df = pd.DataFrame(data['df'])
            if data['weighted_socre'] == -1:  # Skip invalid data
                continue

            _logits = torch.tensor([i for i in df['logits']], dtype=torch.float32)
            human_score = torch.tensor(data['human_score'], dtype=torch.long)
            weighted_score = torch.tensor(df['weighted_score'].to_list(), dtype=torch.float32)

            self.human_score_ls.append(human_score)
            self.logits_ls.append(_logits)
            self.weighted_score_ls.append(weighted_score)

    def __len__(self):
        return len(self.logits_ls)

    def __getitem__(self, idx):
        return self.logits_ls[idx], self.human_score_ls[idx]


def optimize_layer_weights(data_path, num_epochs=2, lr=0.01, min_lr=1e-3, batch_size=8,seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Load data
    dataset = CustomDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    L = len(dataset.logits_ls[0])  
    weights = torch.nn.Parameter(torch.cat([torch.zeros(L - 1)]).to(device), requires_grad=True)
    alpha = torch.nn.Parameter(torch.tensor([0.5]).to(device), requires_grad=True)
    #weights = torch.nn.Parameter(torch.cat([torch.zeros(L)]), requires_grad=True)

    optimizer = optim.Adam([weights,alpha], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=min_lr)

    value_tensor = torch.arange(1, 6, dtype=torch.float32, device=weights.device)
    all_step_losses = []  
    global_step = 0  

    for epoch in trange(num_epochs):
        sig_alpha = torch.sigmoid(alpha)
        total_loss = 0
        for batch_logits, batch_targets in dataloader:
            batch_logits = batch_logits.to(device)[:,:-1,:]
            batch_targets = batch_targets.to(torch.long).to(device)

            
            ce_batch_targets = batch_targets - 1
            mse_batch_targets = batch_targets.to(torch.float32)

            normalized_weights = torch.softmax(weights, dim=0).to(device)

            
            weighted_sum = torch.zeros_like(batch_logits[:, 0])
            for l in range(L-1):
                    weighted_sum += normalized_weights[l] * (batch_logits[:, l])
                   
            predictions = weighted_sum


            # 在循环内部
            predictions_mse = (weighted_sum.softmax(dim=-1) * value_tensor).sum(dim=-1)

            
            loss_ce = nn.CrossEntropyLoss()
            loss1 = loss_ce(predictions, ce_batch_targets)
            loss_mse = nn.MSELoss()
            loss2= loss_mse(predictions_mse, mse_batch_targets)
            loss = sig_alpha*loss1 + (1-sig_alpha)*loss2
            #loss = loss1

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Record the loss for this step
            all_step_losses.append((global_step, loss.item()))
            global_step += 1

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # steps, losses = zip(*all_step_losses)  
    # plt.figure(figsize=(10, 6))
    # plt.plot(steps, losses, marker='.', linestyle='-', color='b', alpha=0.7)
    # plt.title("Training Loss Over Steps")
    # plt.xlabel("Step")
    # plt.ylabel("Loss")
    # plt.grid(True)
    # plt.savefig("loss-step-figure.png")  
    # plt.show()
    return torch.softmax(weights, dim=0).detach()