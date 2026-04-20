### in sample_train_2 we use the artificially imbalanced MNIST training-categorical-classification

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import json
import time
import subprocess
import random
from pprint import pprint
from model_arch.simple_nn import SimpleNN
from collections import Counter
import pickle as pkl

def train_imbalanced_mnist():
    print('Training-Pipeline-(TP)-2...')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    print('MNIST-training-load')    
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    print('MNIST-testing-load')    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    print(len(full_train_dataset), len(test_dataset))
    
    # ARTIFICIAL Imbalance the dataset
    # Keep 100% of '0's, but only 2% of digits '1' through '9'
    skewed_indices =[]
    for idx, (_, label) in enumerate(full_train_dataset):
        if label == 0:
            skewed_indices.append(idx)
        elif random.random() < 0.005:
            skewed_indices.append(idx)
            
    train_dataset = Subset(full_train_dataset, skewed_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    train_label_counter = Counter()
    test_label_counter = Counter()

    for batch in train_loader:
        _, labels = batch   # typical format
        train_label_counter.update(labels.tolist())

    for batch in test_loader:
        _, labels = batch   # typical format
        test_label_counter.update(labels.tolist())

    print('Saved / Train-Label-Distribution: ')
    pprint(train_label_counter)
    with open('./data/train_dist.pkl', 'wb') as f:
        pkl.dump(train_label_counter, f)
    f.close()
    
    print('Saved / Test-Label-Distribution: ')
    pprint(test_label_counter)
    with open('./data/test_dist.pkl', 'wb') as f:
        pkl.dump(test_label_counter, f)
    f.close()
    
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logs = {"run_info": "MNIST Categorical Classification", "epochs":[]}
    epochs_len = 15
    
    for epoch in range(1, epochs_len + 1):  
        model.train()
        train_loss = 0.0
        total_norm = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            ## L2 grad-norm = sqrt(sum(grad^2)) -> more outlier sensitive gradient values
            batch_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.norm(2)
                    batch_norm += param_norm.item() ** 2
            total_norm += batch_norm ** 0.5
            
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            
        train_loss /= len(train_loader.dataset)
        avg_grad_norm = total_norm / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                val_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        val_loss /= len(test_loader.dataset)
        val_acc = correct / len(test_loader.dataset)

        logs["epochs"].append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "avg_grad_norm" : round(avg_grad_norm, 4)
        })
        print(f"Epoch {epoch + 1}/{epochs_len + 1} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    with open("model/logs/training_logs.json", "w") as f:
        json.dump(logs, f, indent=4)
    torch.save(model.state_dict(), "model/models_save/model.pth")
    print("\nTraining Complete. Saved 'model.pth' and 'training_logs.json'.")
    
if __name__ == "__main__":
    train_imbalanced_mnist()