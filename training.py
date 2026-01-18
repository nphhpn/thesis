import csv
import time
import datetime
import numpy as np

import torch


def evaluate(dataloader, model, loss_fn, metrics={}):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    for metric in metrics.values():
        metric.reset()

    with torch.no_grad():
        total_loss = 0
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            logits = model(data)

            loss = loss_fn(logits, label)
            total_loss += loss.item() * len(data)

            probs = torch.sigmoid(logits)
            for metric in metrics.values():
                metric.update(probs, label)

    return total_loss / len(dataloader.dataset), { key: metric.compute() for key, metric in metrics.items() }


def train_one_epoch(dataloader, model, loss_fn, optimizer, scheduler=None, log_timedelta=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    total_loss = 0
    recent_loss = 0
    recent_count = 0
    num_batches = len(dataloader)
    log_timestamp = time.time() + log_timedelta

    for batch, (data, label) in enumerate(dataloader, 1):
        data = data.to(device)
        label = label.to(device)
        logits = model(data)

        loss = loss_fn(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()

        total_loss += loss.item() * len(data)
        recent_loss += loss.item() * len(data)
        recent_count += len(data)

        if log_timedelta > 0 and (time.time() > log_timestamp or batch == num_batches):
            log_timestamp = time.time() + log_timedelta
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} batch {batch}/{num_batches}, loss {recent_loss / recent_count}")
            recent_loss = 0
            recent_count = 0
    return total_loss / len(dataloader.dataset)

    
def get_warmup_decay_scheduler(optimizer, dataloader, num_epochs, warmup_ratio=0.1, final_lr=0.01):
    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    decay_steps = total_steps - warmup_steps
    def lambda_lr(current_steps):
        if current_steps < warmup_steps:
            return current_steps / warmup_steps
        return (total_steps - current_steps) / decay_steps * (1 - final_lr) + final_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)


def train_loop(dataloaders, model, loss_fn, optimizer, log_timedelta=30, metrics={}, num_epochs=100, directory=".", early_stop=10, best_metric="loss"):
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1, 5*len(dataloaders[0]))

    best = -np.inf
    best_epoch = 0
    with open(f"{directory}/metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss"] + list(metrics.keys()))
    
    for epoch in range(1, num_epochs+1):
        if log_timedelta > 0:
            print(f"============ EPOCH {epoch} ============")
        train_loss = train_one_epoch(dataloaders[0], model, loss_fn, optimizer, warmup_scheduler, log_timedelta)

        val_loss, computed_metrics = evaluate(dataloaders[1], model, loss_fn, metrics)
        with open(f"{directory}/metrics.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, val_loss] + list(computed_metrics.values()))
        
        if log_timedelta > 0:
            print("----------")
            print(f"val loss: {val_loss}")
            for key, metric in computed_metrics.items():
                print(f"{key}: {metric}")
        
        current = -val_loss if best_metric == "loss" else computed_metrics[best_metric]
        if epoch == 5:
            warmup_scheduler = None

        if current > best:
            best = current
            best_epoch = epoch
            torch.save(model.state_dict(), f"{directory}/best.pt")
        elif early_stop > 0 and epoch - best_epoch >= early_stop:
            break
        elif epoch - best_epoch >= 5:
            if log_timedelta > 0:
                print("Plateau detected, load best model and reduce lr.")
            model.load_state_dict(torch.load(f"{directory}/best.pt"))
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] / 10