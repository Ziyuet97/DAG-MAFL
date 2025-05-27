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

# 配置参数
NUM_CLIENTS = 10
SELECT_CLIENTS = 5
NUM_EPOCHS = 2
NUM_ROUNDS = 100  # 100轮联邦学习
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIRICHLET_ALPHA = 0.01  # Dirichlet分布的alpha参数，越小越不均衡

# 全局节点参数
GLOBAL_NODE_ID = -2  # 使用-2作为全局节点的ID

# 为CIFAR-10更新模型定义
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
    def __init__(self, dataset, target_classes):
        self.dataset = dataset
        self.target_classes = target_classes
        self.indices = []
        
        # 找出所有符合目标标签的索引
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label in target_classes:
                self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

# 数据加载 - 更新为CIFAR-10
def load_data_dirichlet():
    # CIFAR-10预处理
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
    
    # 为每个节点分配主要关注的两个类别
    node_primary_classes = {}
    for i in range(NUM_CLIENTS):
        # 每个节点主要关注的类别是当前索引和下一个索引对应的类别
        # 使用模运算确保类别在0-9范围内
        class1 = i % 10
        class2 = (i + 1) % 10
        node_primary_classes[i] = [class1, class2]
    
    # 收集每个类别对应的索引
    class_indices = {c: [] for c in range(10)}
    for idx, (_, label) in enumerate(train_set):
        class_indices[label].append(idx)
    
    # 使用Dirichlet分布分配数据
    node_to_indices = {i: [] for i in range(NUM_CLIENTS)}
    
    for class_label in range(10):
        indices = class_indices[class_label]
        
        # 创建Dirichlet分布权重
        # 设置主要负责该类别的节点权重较高
        dirichlet_params = np.ones(NUM_CLIENTS) * DIRICHLET_ALPHA
        for node_id, primary_classes in node_primary_classes.items():
            if class_label in primary_classes:
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
    client_test_datasets = []  # 用于本地测试的数据集，只包含主要类别
    
    for i in range(NUM_CLIENTS):
        indices = node_to_indices[i]
        random.shuffle(indices)
        client_dataset = Subset(train_set, indices)
        client_datasets.append(client_dataset)
        
        # 为每个客户端创建一个只包含其主要类别的测试数据集
        primary_classes = node_primary_classes[i]
        filtered_test_dataset = FilteredDataset(test_set, primary_classes)
        client_test_datasets.append(filtered_test_dataset)
    
    # 统计验证数据分布
    for i, ds in enumerate(client_datasets):
        primary_classes = node_primary_classes[i]
        labels = []
        for idx in ds.indices:
            _, label = train_set[idx]
            labels.append(label)
        
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        print(f"\nClient {i} data distribution: {distribution}")
        print(f"Primary classes: {primary_classes}")
        
        # 计算主要类别的占比
        primary_count = sum(counts[np.where(unique == c)[0][0]] if c in unique else 0 for c in primary_classes)
        total_count = sum(counts)
        primary_ratio = primary_count / total_count if total_count > 0 else 0
        print(f"Primary classes ratio: {primary_ratio:.2f}")
    
    # 创建数据加载器
    train_loaders = [
        DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        for ds in client_datasets
    ]
    
    client_test_loaders = [
        DataLoader(ds, batch_size=512, pin_memory=True)
        for ds in client_test_datasets
    ]
    
    test_loader = DataLoader(test_set, batch_size=512, pin_memory=True)
    
    return train_loaders, client_test_loaders, test_loader, node_primary_classes

# 训练函数
def train(model, train_loader, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

# 测试函数
def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy

# 模型参数聚合函数
def federated_avg(models, client_data_sizes):
    """
    基于客户端数据大小的联邦平均算法
    """
    # 获取模型参数
    global_model = CIFAR10CNN().to(DEVICE)
    global_dict = global_model.state_dict()
    
    total_size = sum(client_data_sizes)
    
    # 按照客户端数据量加权平均所有模型参数
    for k in global_dict.keys():
        global_dict[k] = torch.stack([models[i].state_dict()[k].float() * (client_data_sizes[i] / total_size) 
                                    for i in range(len(models))], 0).sum(0)
    
    # 加载回全局模型
    global_model.load_state_dict(global_dict)
    return global_model

def main():
    # 加载数据
    train_loaders, client_test_loaders, global_test_loader, node_primary_classes = load_data_dirichlet()
    
    # 获取每个客户端的数据量
    client_data_sizes = [len(loader.dataset) for loader in train_loaders]
    
    # 初始化全局模型
    global_model = CIFAR10CNN().to(DEVICE)
    
    # 评估初始模型
    global_test_loss, global_accuracy = test(global_model, global_test_loader, DEVICE)
    print(f"Initial Global Model - Test Loss: {global_test_loss:.4f}, Accuracy: {global_accuracy:.2f}%")
    
    # 存储结果用于绘图
    global_accuracies = [global_accuracy]
    local_avg_accuracies = []
    
    # 存储每个节点的最新模型
    node_models = [CIFAR10CNN().to(DEVICE) for _ in range(NUM_CLIENTS)]
    for model in node_models:
        model.load_state_dict(global_model.state_dict())
    
    # 联邦学习主循环
    for round_idx in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_idx}/{NUM_ROUNDS} ---")
        
        # 选择参与训练的客户端
        selected_clients = random.sample(range(NUM_CLIENTS), SELECT_CLIENTS)
        print(f"Selected clients: {selected_clients}")
        
        # 客户端模型列表
        client_models = []
        selected_data_sizes = []
        
        # 客户端本地训练
        for client_idx in selected_clients:
            # 初始化客户端模型（从全局模型复制参数）
            client_model = CIFAR10CNN().to(DEVICE)
            client_model.load_state_dict(global_model.state_dict())
            
            # 设置优化器 - 调整学习率以适应CIFAR-10
            optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            
            # 本地训练
            for epoch in range(NUM_EPOCHS):
                train(client_model, train_loaders[client_idx], optimizer, DEVICE)
            
            # 更新该节点的最新模型
            node_models[client_idx].load_state_dict(client_model.state_dict())
            
            # 保存客户端模型和数据大小，用于后续聚合
            client_models.append(client_model)
            selected_data_sizes.append(client_data_sizes[client_idx])
        
        # 计算所有节点的平均准确度
        total_acc = 0.0
        for node_id in range(NUM_CLIENTS):
            _, node_accuracy = test(node_models[node_id], client_test_loaders[node_id], DEVICE)
            total_acc += node_accuracy
        avg_accuracy = total_acc / NUM_CLIENTS
        local_avg_accuracies.append(avg_accuracy)
        print(f"Round {round_idx} - All Nodes Average Accuracy: {avg_accuracy:.2f}%")
        
        # 使用FedAvg聚合模型参数
        global_model = federated_avg(client_models, selected_data_sizes)
        
        # 在全局测试数据上评估聚合后的模型
        global_test_loss, global_accuracy = test(global_model, global_test_loader, DEVICE)
        global_accuracies.append(global_accuracy)
        print(f"Round {round_idx} - Global Model Test Loss: {global_test_loss:.4f}, Accuracy: {global_accuracy:.2f}%")
    
    print("local_avg_accuracies:", local_avg_accuracies)
    print("global_accuracies:", global_accuracies)
    
    # 绘制准确度曲线
    plt.figure(figsize=(10, 6))
    
    # 绘制全局模型准确度
    plt.plot(range(NUM_ROUNDS + 1), global_accuracies, label='Global Model Accuracy', marker='o')
    
    # 绘制本地平均准确度
    plt.plot(range(1, NUM_ROUNDS + 1), local_avg_accuracies, label='All Nodes Average Accuracy', marker='x')
    
    plt.title('FedAvg Accuracy on CIFAR-10 over Training Rounds')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('fedavg_cifar10_accuracy.png')
    plt.show()

if __name__ == "__main__":
    main()