import networkx as nx
import random
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from collections import defaultdict

# 配置参数
NUM_CLIENTS = 10
SELECT_CLIENTS = 5
NUM_EPOCHS = 2
NUM_ROUNDS = 100
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIRICHLET_ALPHA = 0.01
GLOBAL_AGGREGATION_INTERVAL = 5

# 模型定义
class FastMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class FilteredDataset(Dataset):
    def __init__(self, dataset, target_digits):
        self.dataset = dataset
        self.target_digits = target_digits
        self.indices = [i for i, (_, label) in enumerate(dataset) if label in target_digits]

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def load_data_dirichlet():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    
    node_primary_digits = {i: [i%10, (i+1)%10] for i in range(NUM_CLIENTS)}
    
    digit_indices = {d: [] for d in range(10)}
    for idx, (_, label) in enumerate(train_set):
        digit_indices[label].append(idx)
    
    node_to_indices = {i: [] for i in range(NUM_CLIENTS)}
    
    for digit in range(10):
        indices = digit_indices[digit]
        dirichlet_params = np.ones(NUM_CLIENTS) * DIRICHLET_ALPHA
        for node_id, primary_digits in node_primary_digits.items():
            if digit in primary_digits:
                dirichlet_params[node_id] = 10.0
        
        proportions = np.random.dirichlet(dirichlet_params)
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)
        start_idx = 0
        
        for node_id in range(NUM_CLIENTS):
            end_idx = proportions[node_id] if node_id != NUM_CLIENTS-1 else len(indices)
            selected_indices = indices[start_idx:end_idx]
            node_to_indices[node_id].extend(selected_indices)
            start_idx = end_idx
    
    client_datasets = []
    client_test_datasets = []
    for node_id in range(NUM_CLIENTS):
        node_indices = list(set(node_to_indices[node_id]))
        random.shuffle(node_indices)
        client_datasets.append(Subset(train_set, node_indices))
        client_test_datasets.append(FilteredDataset(test_set, node_primary_digits[node_id]))
    
    train_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]
    client_test_loaders = [DataLoader(ds, batch_size=BATCH_SIZE) for ds in client_test_datasets]
    global_test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    
    return train_loaders, client_test_loaders, global_test_loader, node_primary_digits

class MOCHAClient:
    def __init__(self, client_id, train_loader, test_loader):
        self.client_id = client_id
        self.model = FastMNIST().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.personalized_params = None

    def train(self, global_model, mu=0.1):
        self.model.train()
        global_model.eval()
        global_params = [p.detach().clone() for p in global_model.parameters()]
        
        for _ in range(NUM_EPOCHS):
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                
                reg_loss = 0.0
                for local_p, global_p in zip(self.model.parameters(), global_params):
                    reg_loss += torch.norm(local_p - global_p, p=2)
                loss += mu * reg_loss
                
                loss.backward()
                self.optimizer.step()
        
        self.personalized_params = [p.detach().clone() for p in self.model.parameters()]
        return len(self.train_loader.dataset)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        return test_loss, accuracy

# 初始化
train_loaders, client_test_loaders, global_test_loader, _ = load_data_dirichlet()
clients = [MOCHAClient(i, train_loaders[i], client_test_loaders[i]) for i in range(NUM_CLIENTS)]
global_model = FastMNIST().to(DEVICE)
global_optimizer = optim.Adam(global_model.parameters(), lr=1e-3)

# 训练循环
transaction_count = 0
global_accuracy_history = []
local_accuracy_history = defaultdict(list)
recent_selected_for_aggregation = []

for round in range(NUM_ROUNDS):
    selected_clients = random.sample(clients, SELECT_CLIENTS)
    
    # 分发全局参数
    global_params = [p.detach().clone() for p in global_model.parameters()]
    for client in selected_clients:
        for p, gp in zip(client.model.parameters(), global_params):
            p.data.copy_(gp)
    
    # 本地训练
    for client in selected_clients:
        client.train(global_model)
        transaction_count += 1
        recent_selected_for_aggregation.append(client)
        
        if transaction_count % GLOBAL_AGGREGATION_INTERVAL == 0:
            avg_params = []
            for i in range(len(global_params)):
                layer_params = torch.stack([c.personalized_params[i] for c in recent_selected_for_aggregation], dim=0)
                avg_layer = layer_params.mean(dim=0)
                avg_params.append(avg_layer)
            
            with torch.no_grad():
                for global_p, avg_p in zip(global_model.parameters(), avg_params):
                    global_p.data.copy_(avg_p)
            
            recent_selected_for_aggregation = []
            
            # 全局测试
            global_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in global_test_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = global_model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            global_acc = correct / total
            global_accuracy_history.append(global_acc)
            print(f"Round {round}, Global Acc: {global_acc:.4f}")
    
    # 记录本地准确率
    for client in clients:
        _, acc = client.test()
        local_accuracy_history[client.client_id].append(acc)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(global_accuracy_history, label='Global Model')
for client_id, accs in local_accuracy_history.items():
    plt.plot(accs, label=f'Client {client_id}', alpha=0.5)
plt.xlabel('Communication Rounds')
plt.ylabel('Accuracy')
plt.title('MOCHA Performance on MNIST')
plt.legend()
plt.grid()
plt.show()

print(local_accuracy_history)
print(global_accuracy_history)
