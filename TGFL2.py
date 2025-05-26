import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import networkx as nx

# 配置参数
NUM_CLIENTS = 10
NUM_EPOCHS = 2
NUM_ROUNDS = 100  # 运行100轮
BATCH_SIZE = 128
DIRICHLET_ALPHA = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FastMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, allowed_labels):
        self.dataset = dataset
        self.indices = [i for i, (_, label) in enumerate(dataset) if label in allowed_labels]
    
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    
    def __len__(self):
        return len(self.indices)

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

def client_update(model, loader, optimizer):
    model.train()
    model.to(DEVICE)
    for _ in range(NUM_EPOCHS):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(X)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
    return model.cpu().state_dict()

def federated_average(params_list):
    avg_params = {}
    for key in params_list[0].keys():
        avg_params[key] = torch.stack([p[key].float() for p in params_list], dim=0).mean(dim=0)
    return avg_params

def evaluate(model, test_loader):
    model.to(DEVICE)
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X).argmax(dim=1)
            correct += pred.eq(y).sum().item()
    return 100. * correct / len(test_loader.dataset)

class EnhancedDAG:
    def __init__(self, num_nodes=10):
        self.num_nodes = num_nodes
        self.graph = nx.DiGraph()
        self.current_id = 0
        self.tx_params = []
        self.latest_tx_per_node = {i: 0 for i in range(num_nodes)}
        self._init_genesis()

    def _init_genesis(self):
        initial_model = FastMNIST()
        self.graph.add_node(self.current_id, node_id=-1)
        self.tx_params.append(initial_model.state_dict())
        self.current_id += 1

    def generate_transaction(self):
        current_nodes = list(self.graph.nodes)
        selected_nodes = []
        
        for node_id in range(self.num_nodes):
            if random.random() < 0.5:
                candidates = sorted([n for n in current_nodes if n != self.current_id], reverse=True)[:10]
                acc_scores = []
                
                for p in candidates:
                    if p >= len(self.tx_params):
                        continue
                    model = FastMNIST()
                    model.load_state_dict(self.tx_params[p])
                    acc = evaluate(model, client_test_loaders[node_id])
                    acc_scores.append((p, acc))
                
                sorted_acc = sorted(acc_scores, key=lambda x: x[1], reverse=True)
                parents = [p for p, _ in sorted_acc[:2]]
                
                if parents:
                    selected_nodes.append((node_id, parents))
        
        for node_id, parents in selected_nodes:
            parent_params = [self.tx_params[p] for p in parents if p < len(self.tx_params)]
            aggregated_params = federated_average(parent_params) if parent_params else self.tx_params[0]
            
            local_model = FastMNIST()
            local_model.load_state_dict(aggregated_params)
            optimizer = optim.SGD(local_model.parameters(), lr=0.001, momentum=0.9)
            
            updated_params = client_update(local_model, train_loaders[node_id], optimizer)
            new_tx_id = self.current_id
            
            self.graph.add_node(new_tx_id, node_id=node_id)
            self.tx_params.append(updated_params)
            self.latest_tx_per_node[node_id] = new_tx_id
            self.current_id += 1

# 初始化系统和数据
torch.manual_seed(1)
train_loaders, client_test_loaders, global_test_loader, _ = load_data_dirichlet()
acc_history = []

# 创建DAG实例并运行
sim = EnhancedDAG(num_nodes=10)
for round in range(NUM_ROUNDS):
    sim.generate_transaction()
    
    # 计算平均准确度
    total_acc = 0.0
    for node_id in range(10):
        tx_id = sim.latest_tx_per_node[node_id]
        model = FastMNIST()
        model.load_state_dict(sim.tx_params[tx_id])
        acc = evaluate(model, client_test_loaders[node_id])
        total_acc += acc
    acc_history.append(total_acc/10)
    
    print(f"Round {round+1}/{NUM_ROUNDS} - Avg Accuracy: {total_acc/10:.2f}%")
    print(acc_history)