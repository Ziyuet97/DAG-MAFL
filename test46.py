import random
import time
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, Dataset

# 配置参数
NUM_CLIENTS = 10
SELECT_CLIENTS = 5
NUM_EPOCHS = 2
NUM_ROUNDS = 100
BATCH_SIZE = 128
Recent_TX = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K_MODELS = 5
N_RECENT_MODELS = 10
DIRICHLET_ALPHA = 0.01
SPARSIFICATION_RATIO = 0.3

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
    
    train_set = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    
    node_primary_digits = {i: [i%10, (i+1)%10] for i in range(NUM_CLIENTS)}
    digit_indices = {d: [] for d in range(10)}
    for idx, (_, label) in enumerate(train_set):
        digit_indices[label].append(idx)
    
    node_to_indices = {i: [] for i in range(NUM_CLIENTS)}
    for digit in range(10):
        indices = digit_indices[digit]
        dirichlet_params = np.ones(NUM_CLIENTS) * DIRICHLET_ALPHA
        for node_id, digits in node_primary_digits.items():
            if digit in digits:
                dirichlet_params[node_id] = 10.0
        
        proportions = np.random.dirichlet(dirichlet_params)
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)
        start_idx = 0
        for node_id in range(NUM_CLIENTS):
            end_idx = proportions[node_id] if node_id != NUM_CLIENTS-1 else len(indices)
            node_to_indices[node_id].extend(indices[start_idx:end_idx])
            start_idx = end_idx
    
    client_datasets = [Subset(train_set, indices) for indices in node_to_indices.values()]
    client_test_datasets = [FilteredDataset(test_set, node_primary_digits[i]) for i in range(NUM_CLIENTS)]
    
    train_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]
    client_test_loaders = [DataLoader(ds, batch_size=512) for ds in client_test_datasets]
    test_loader = DataLoader(test_set, batch_size=512)
    
    return train_loaders, client_test_loaders, test_loader, node_primary_digits

def client_update(model, loader, optimizer):
    model.train().to(DEVICE)
    subset_size = max(1, int(len(loader.dataset)*1))
    indices = random.sample(range(len(loader.dataset)), subset_size)
    temp_loader = DataLoader(Subset(loader.dataset, indices), batch_size=BATCH_SIZE, shuffle=True)
    
    for _ in range(NUM_EPOCHS):
        for X, y in temp_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            optimizer.step()
    return model.cpu().state_dict()

def get_model_importance_score(params):
    return sum(torch.sum(torch.abs(v)).item() for k,v in params.items() if 'weight' in k or 'bias' in k)

def federated_average(params_list):
    return {k: torch.stack([p[k].float() for p in params_list]).mean(0) for k in params_list[0].keys()}

class EnhancedDAG:
    def __init__(self, num_nodes=10, node_primary_digits=None):
        self.num_nodes = num_nodes
        self.graph = nx.DiGraph()
        self.current_id = 0
        self.tx_params = []
        self.tx_accuracies = []
        self.tx_global_accuracies = []
        self.tx_importance_scores = []
        self.node_primary_digits = node_primary_digits
        self._init_genesis()

    def _init_genesis(self):
        initial_model = FastMNIST()
        self.graph.add_node(self.current_id, node_id=-1)
        self.tx_params.append(initial_model.state_dict())
        initial_acc = self.evaluate_model_accuracy(initial_model.state_dict(), use_global=True)
        self.tx_accuracies.append(0.0)
        self.tx_global_accuracies.append(initial_acc)
        self.tx_importance_scores.append(0.0)
        self.current_id += 1

    def evaluate_model_accuracy(self, model_params, node_id=None, use_global=False):
        """统一的模型评估方法"""
        model = FastMNIST()
        model.load_state_dict(model_params)
        model.eval().to(DEVICE)
        
        # 选择测试集
        if use_global:
            loader = test_loader
        elif node_id is not None and node_id < len(client_test_loaders):
            loader = client_test_loaders[node_id]
        else:
            loader = test_loader
        
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = model(X)
                pred = outputs.argmax(dim=1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        
        return 100.0 * correct / total if total > 0 else 0.0

    def generate_round_transactions(self):
        current_nodes = list(self.graph.nodes)
        selected_clients = random.sample(range(self.num_nodes), SELECT_CLIENTS)
        new_tx_ids = []
        
        for node_id in selected_clients:
            parents = []
            recent_txs = sorted([n for n in current_nodes if n !=0], reverse=True)[:N_RECENT_MODELS]
            tx_acc_pairs = [(tx, self.tx_global_accuracies[tx] if tx<len(self.tx_global_accuracies) else 0) 
                           for tx in recent_txs]
            parents = [tx for tx, _ in sorted(tx_acc_pairs, key=lambda x: x[1], reverse=True)[:3]]
            if not parents:
                parents = [0]

            parent_params = [self.tx_params[p] for p in parents if p < len(self.tx_params)]
            aggregated_params = federated_average(parent_params) if parent_params else self.tx_params[0]
            
            model = FastMNIST()
            model.load_state_dict(aggregated_params)
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            updated_params = client_update(model, train_loaders[node_id], optimizer)
            
            new_tx_id = self.current_id
            self.graph.add_node(new_tx_id, node_id=node_id)
            self.graph.add_edges_from([(p, new_tx_id) for p in parents])
            
            self.tx_params.append(updated_params)
            self.tx_accuracies.append(self.evaluate_model_accuracy(updated_params, node_id))
            self.tx_global_accuracies.append(self.evaluate_model_accuracy(updated_params, use_global=True))
            self.tx_importance_scores.append(get_model_importance_score(updated_params))
            
            new_tx_ids.append(new_tx_id)
            self.current_id += 1
        
        return new_tx_ids

    def aggregate_models(self):
        if len(self.graph) <= 1:
            return 0.0, None, []

        recent_txs = sorted([n for n in self.graph.nodes if n !=0])[-N_RECENT_MODELS:]
        importance_scores = [(tx, get_model_importance_score(self.tx_params[tx])) for tx in recent_txs]
        selected_txs = [tx for tx, _ in sorted(importance_scores, key=lambda x: x[1], reverse=True)[:TOP_K_MODELS]]
        
        if not selected_txs:
            return 0.0, None, []
        
        global_params = federated_average([self.tx_params[tx] for tx in selected_txs])
        acc = self.evaluate_model_accuracy(global_params, use_global=True)
        return acc, global_params, selected_txs

# 初始化系统和数据
train_loaders, client_test_loaders, test_loader, node_primary_digits = load_data_dirichlet()

# 初始化全局模型并评估
initial_model = FastMNIST().to(DEVICE)
initial_acc = EnhancedDAG().evaluate_model_accuracy(initial_model.state_dict(), use_global=True)
print(f"Initial Global Model Accuracy: {initial_acc:.2f}%")

# 运行联邦学习
sim = EnhancedDAG(num_nodes=NUM_CLIENTS, node_primary_digits=node_primary_digits)
global_accuracies = []
local_avg_accuracies = []

for round in range(NUM_ROUNDS):
    start_time = time.time()
    round_local_acc = []
    
    # 生成本轮交易
    new_tx_ids = sim.generate_round_transactions()
    
    # 收集本地精度
    for tx_id in new_tx_ids:
        if tx_id < len(sim.tx_accuracies):
            round_local_acc.append(sim.tx_accuracies[tx_id])
    
    # 计算本轮本地平均精度
    if len(round_local_acc) > 0:
        avg_local = np.mean(round_local_acc)
    else:
        avg_local = 0.0
    local_avg_accuracies.append(avg_local)
    
    # 聚合模型并记录全局精度
    acc, global_params, selected_txs = sim.aggregate_models()
    if global_params is not None:
        global_accuracies.append(acc)
        print(f"Round {round+1:02d} | Global: {acc:.2f}% | Local Avg: {avg_local:.2f}% | Time: {time.time()-start_time:.1f}s")
    else:
        print(f"Round {round+1:02d} | No aggregation | Local Avg: {avg_local:.2f}% | Time: {time.time()-start_time:.1f}s")

# 最终结果输出
print("\n=== Final Results ===")
print("Global Accuracies:")
print([f"{acc:.2f}%" for acc in global_accuracies])

print("\nLocal Average Accuracies:")
print([f"{acc:.2f}%" for acc in local_avg_accuracies])

# 可视化精度趋势
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(global_accuracies, label='Global Accuracy')
plt.plot(local_avg_accuracies, label='Local Average Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('Federated Learning Performance')
plt.legend()
plt.grid(True)
plt.show()