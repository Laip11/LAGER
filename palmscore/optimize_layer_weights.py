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


def optimize_layer_weights(data_path, loss_fn, num_epochs=2, lr=0.01, min_lr=1e-3, batch_size=8,seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Load data
    dataset = CustomDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    L = len(dataset.logits_ls[0])  # Number of layers in logits
    weights = torch.nn.Parameter(torch.cat([torch.zeros(L - 1), torch.tensor([1.0])]), requires_grad=True)
    #weights = torch.nn.Parameter(torch.cat([torch.zeros(L)]), requires_grad=True)

    optimizer = optim.Adam([weights], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=min_lr)

    all_step_losses = []  # Store loss for each step
    global_step = 0  # Global step counter

    for epoch in trange(num_epochs):
        total_loss = 0
        for batch_logits, batch_targets in dataloader:
            batch_logits = batch_logits.to(torch.float32)
            batch_targets = batch_targets.to(torch.long)

            # Adjust target for CrossEntropyLoss
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                batch_targets = batch_targets - 1

            normalized_weights = torch.softmax(weights, dim=0)

            # Compute weighted sum for each sample in the batch
            weighted_sum = torch.zeros_like(batch_logits[:, 0])
            for l in range(L):
                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    weighted_sum += normalized_weights[l] * (batch_logits[:, l])
                else:
                    weighted_sum += normalized_weights[l] * (batch_logits[:, l] * torch.tensor([1, 2, 3, 4, 5])).sum(dim=1)

            predictions = weighted_sum
            loss = loss_fn(predictions, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record the loss for this step
            all_step_losses.append((global_step, loss.item()))
            global_step += 1

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)


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