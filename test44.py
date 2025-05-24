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
SELECT_CLIENTS = 5  # 将SELECT_CLIENTS从10改为5

# 在训练循环外初始化用于聚合的客户端列表
recent_selected_for_aggregation = []
NUM_EPOCHS = 2
NUM_ROUNDS = 100  # 增加到100轮
BATCH_SIZE = 128
Recent_TX = 3  # 每个节点引用的交易数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K_MODELS = 5  # 选择前K个模型（基于参数幅度）
N_RECENT_MODELS = 11  # 全局聚合时考虑最近N个模型
DIRICHLET_ALPHA = 0.01  # Dirichlet分布的alpha参数，越小越不均衡
SPARSIFICATION_RATIO = 0.3  # Top-k稀疏化比例，保留参数的30%

# 新增全局节点参数
GLOBAL_NODE_ID = -2  # 使用-2作为全局节点的ID（与普通节点和创世块区分）
GLOBAL_AGGREGATION_INTERVAL = 5  # 每10个交易执行一次全局聚合
N_REFERENCE_CANDIDATES = 6  # 每个节点考虑的候选引用交易数

# 模型定义
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # CIFAR-10: 3通道彩色图像，32x32像素
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: [batch_size, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch_size, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch_size, 64, 8, 8]
        x = self.pool(F.relu(self.conv3(x)))  # -> [batch_size, 64, 4, 4]
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 创建一个支持过滤特定标签的数据集包装器
class FilteredDataset(Dataset):
    def __init__(self, dataset, target_digits):
        self.dataset = dataset
        self.target_digits = target_digits
        self.indices = []
        
        # 找出所有符合目标标签的索引
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label in target_digits:
                self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

# 数据加载
def load_data_dirichlet():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    # 为每个节点分配主要关注的两个数字
    node_primary_digits = {}
    for i in range(NUM_CLIENTS):
        # 每个节点主要关注的数字是当前索引和下一个索引对应的数字
        # 使用模运算确保数字在0-9范围内
        digit1 = i % 10
        digit2 = (i + 1) % 10
        node_primary_digits[i] = [digit1, digit2]
    
    # 收集每个数字对应的索引
    digit_indices = {d: [] for d in range(10)}
    for idx, (_, label) in enumerate(train_set):
        digit_indices[label].append(idx)
    
    # 使用Dirichlet分布分配数据
    node_to_indices = {i: [] for i in range(NUM_CLIENTS)}
    
    for digit in range(10):
        indices = digit_indices[digit]
        
        # 创建Dirichlet分布权重
        # 设置主要负责该数字的节点权重较高
        dirichlet_params = np.ones(NUM_CLIENTS) * DIRICHLET_ALPHA
        for node_id, primary_digits in node_primary_digits.items():
            if digit in primary_digits:
                dirichlet_params[node_id] = 10.0  # 给主要负责的节点设置更高的权重
        
        # 生成Dirichlet分布
        proportions = np.random.dirichlet(dirichlet_params)
        
        # 根据比例分配索引
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)
        start_idx = 0
        
        # 分配数据给各节点
        for node_id in range(NUM_CLIENTS):
            end_idx = proportions[node_id]
            # 确保最后一个节点获取所有剩余数据
            if node_id == NUM_CLIENTS - 1:
                end_idx = len(indices)
            
            # 分配数据
            selected_indices = indices[start_idx:end_idx]
            node_to_indices[node_id].extend(selected_indices)
            start_idx = end_idx
    
    # 创建客户端数据集并打乱顺序
    client_datasets = []
    client_test_datasets = []  # 用于本地测试的数据集，只包含主要数字

    # （续接原有代码）完成客户端数据集创建
    for node_id in range(NUM_CLIENTS):
        # 训练集：包含Dirichlet分配的所有数据
        node_indices = list(set(node_to_indices[node_id]))  # 去重
        random.shuffle(node_indices)
        client_dataset = Subset(train_set, node_indices)
        client_datasets.append(client_dataset)
        
        # 测试集：仅包含主要关注的两个数字
        test_filtered = FilteredDataset(test_set, node_primary_digits[node_id])
        client_test_datasets.append(test_filtered)
    
    return client_datasets, client_test_datasets, test_set

# MOCHA客户端类
class MOCHAClient:
    def __init__(self, client_id, train_dataset, test_dataset):
        self.client_id = client_id
        self.model = CIFAR10CNN().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # MOCHA参数
        self.local_epoch = 0
        self.personalized_params = None
        
    def train(self, global_model, mu=0.1):
        """执行本地训练，加入MOCHA正则项"""
        self.model.train()
        global_model.eval()
        
        # 保存全局模型参数用于正则项计算
        global_params = [p.detach().clone() for p in global_model.parameters()]
        
        for epoch in range(NUM_EPOCHS):
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                
                # MOCHA正则项：与全局模型的L2距离
                reg_loss = 0.0
                for local_p, global_p in zip(self.model.parameters(), global_params):
                    reg_loss += torch.norm(local_p - global_p, p=2)
                loss += mu * reg_loss
                
                loss.backward()
                self.optimizer.step()
        
        # 更新个性化参数
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

# 初始化全局模型和客户端
client_datasets, client_test_datasets, global_test_set = load_data_dirichlet()
clients = [MOCHAClient(i, client_datasets[i], client_test_datasets[i]) for i in range(NUM_CLIENTS)]
global_model = CIFAR10CNN().to(DEVICE)
global_optimizer = optim.Adam(global_model.parameters(), lr=1e-3)

# 全局测试集
global_test_loader = DataLoader(global_test_set, batch_size=BATCH_SIZE)

# 训练循环
transaction_count = 0
global_accuracy_history = []
local_accuracy_history = defaultdict(list)

# 修改后的训练循环部分
for round in range(NUM_ROUNDS):
    # 选择客户端（每次选择5个）
    selected_clients = random.sample(clients, SELECT_CLIENTS)
    
    # 分发全局模型参数
    global_params = [p.detach().clone() for p in global_model.parameters()]
    for client in selected_clients:
        for p, gp in zip(client.model.parameters(), global_params):
            p.data.copy_(gp)
    
    # 客户端本地训练
    for client in selected_clients:
        client.train(global_model)
        transaction_count += 1
        recent_selected_for_aggregation.append(client)  # 记录参与训练的客户端
        
        # 检查是否执行全局聚合
        if transaction_count % GLOBAL_AGGREGATION_INTERVAL == 0:
            # 聚合最近参与训练的客户端参数
            avg_params = []
            for i in range(len(global_params)):
                # 仅使用最近参与训练的客户端进行聚合
                layer_params = torch.stack([c.personalized_params[i] for c in recent_selected_for_aggregation], dim=0)
                avg_layer = layer_params.mean(dim=0)
                avg_params.append(avg_layer)
            
            # 更新全局模型
            with torch.no_grad():
                for global_p, avg_p in zip(global_model.parameters(), avg_params):
                    global_p.data.copy_(avg_p)
            
            # 清空聚合列表
            recent_selected_for_aggregation = []
            
            # 测试全局模型
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

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(global_accuracy_history, label='Global Model')
for client_id, accs in local_accuracy_history.items():
    plt.plot(accs, label=f'Client {client_id}', alpha=0.5)
plt.xlabel('Communication Rounds')
plt.ylabel('Accuracy')
plt.title('MOCHA Performance on CIFAR-10')
plt.legend()
plt.grid()
plt.show()
for client_id, accs in local_accuracy_history.items():
    print(client_id)
    print(accs)
print(global_accuracy_history)
