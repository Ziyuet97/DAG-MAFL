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

# 配置参数
NUM_CLIENTS = 10
SELECT_CLIENTS = 5
NUM_EPOCHS = 2
NUM_ROUNDS = 100
BATCH_SIZE = 128
Recent_TX = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K_MODELS = 5
N_RECENT_MODELS = 11
DIRICHLET_ALPHA = 0.01
SPARSIFICATION_RATIO = 0.3

# 全局节点参数
GLOBAL_NODE_ID = -2
GLOBAL_AGGREGATION_INTERVAL = 10
N_REFERENCE_CANDIDATES = 6

# CIFAR10模型定义（修改点1）
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class FilteredDataset(Dataset):
    def __init__(self, dataset, target_digits):
        self.dataset = dataset
        self.target_digits = target_digits
        self.indices = [i for i, (_, label) in enumerate(dataset) if label in target_digits]

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

# 数据加载修改为CIFAR10（修改点2）
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
    
    # 打印数据分布
    for i, ds in enumerate(client_datasets):
        labels = [train_set[idx][1] for idx in ds.indices]
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        primary_digits = node_primary_digits[i]
        primary_count = sum(counts[np.where(unique == d)[0][0]] if d in unique else 0 for d in primary_digits)
        total_count = sum(counts)
        print(f"Client {i}: Primary digits {primary_digits} ratio {primary_count/total_count:.2f}")
    
    return train_loaders, client_test_loaders, test_loader, node_primary_digits

# 修改客户端训练函数（修改点3）
def client_update(model, loader, optimizer):
    model.train()
    model.to(DEVICE)
    for _ in range(NUM_EPOCHS):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(X)  # 修改输入处理
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
    return model.cpu().state_dict()

# 其余工具函数需要修改输入处理部分
def get_model_importance_score(params):
    total_score = 0
    for key, value in params.items():
        if 'weight' in key or 'bias' in key:
            total_score += torch.sum(torch.abs(value)).item()
    return total_score

def topk_sparsification_fedavg(params_list, k=TOP_K_MODELS):
    if len(params_list) <= k:
        return federated_average(params_list)
    
    importance_scores = [get_model_importance_score(params) for params in params_list]
    topk_indices = np.argsort(importance_scores)[-k:]
    topk_params = [params_list[i] for i in topk_indices]
    return federated_average(topk_params)

def federated_average(params_list):
    avg_params = {}
    for key in params_list[0].keys():
        avg_params[key] = torch.stack([p[key].float() for p in params_list], dim=0).mean(dim=0)
    return avg_params

def evaluate(model, test_loader):
    model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = model(X)  # 修改输入处理
            pred = output.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += len(y)
    return 100. * correct / total if total > 0 else 0









# DAG区块链类需要修改模型初始化（修改点4）
class DAGBlockchainFL:
    def __init__(self, num_nodes=10, node_primary_digits=None):
        self.num_nodes = num_nodes
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.current_id = 0
        self.tx_params = []
        self.tx_accuracies = []
        self.tx_global_accuracies = []
        self.tx_importance_scores = []
        self.node_primary_digits = node_primary_digits
        self.round = 0
        self._init_genesis()
        self.last_global_tx = 0
        self.latest_tx_per_node = {i: 0 for i in range(num_nodes)}
        
        self.global_acc_history = []
        self.round_history = []
        self.local_acc_history = []
        self.round_transactions = {}
        self.nodes_active_in_round = {}

    def _init_genesis(self):
        initial_model = CIFAR10CNN()  # 修改为CIFAR10模型
        self.graph.add_node(self.current_id, node_id=-1)
        self.node_positions[self.current_id] = (0, 0)
        self.tx_params.append(initial_model.state_dict())
        self.tx_accuracies.append(0.0)
        self.tx_global_accuracies.append(0.0)
        self.tx_importance_scores.append(0.0)
        self.current_id += 1

    def evaluate_model_accuracy(self, model_params, node_id=None, use_global=False):
        model = CIFAR10CNN()  # 修改为CIFAR10模型
        model.load_state_dict(model_params)
        
        if use_global or node_id is None:
            return evaluate(model, test_loader)
        elif node_id < len(client_test_loaders):
            return evaluate(model, client_test_loaders[node_id])
        else:
            return evaluate(model, test_loader)
    
    def evaluate_model_on_primary_digits(self, model_params, node_id):
        if node_id >= len(client_test_loaders):
            return 0.0
            
        model = CIFAR10CNN()  # 修改为CIFAR10模型
        model.load_state_dict(model_params)
        return evaluate(model, client_test_loaders[node_id])

    def topk_sparsification_aggregation(self, create_global_tx=False):
        """使用两个全局节点之间的全部交易进行联邦平均，每个节点只保留最新交易"""
        if len(self.graph) <= 1:  # 只有创世块
            print("No transactions to aggregate!")
            return 0.0, None, None

        # 获取所有普通交易（排除创世块和之前的全局节点）
        all_txs = [n for n in self.graph.nodes 
                if n != 0 and self.graph.nodes[n].get('node_id') != GLOBAL_NODE_ID]
        
        # 获取在上次全局交易之后产生的交易
        start_tx = self.last_global_tx + 1
        candidate_txs = [tx for tx in all_txs if tx >= start_tx]
        
        # 如果范围为空则使用最近的交易
        if not candidate_txs:
            candidate_txs = sorted(all_txs)[-1:]  # 使用最近1个交易

        print(f"\n📊 Found {len(candidate_txs)} candidate transactions between TX{start_tx}-{max(candidate_txs)}")

        # 按节点分组，保留每个节点最新的交易
        node_tx_dict = {}
        for tx in candidate_txs:
            node_id = self.graph.nodes[tx].get('node_id', -1)
            # 排除创世块和全局节点（理论上不会存在）
            if node_id >= 0:  
                if node_id not in node_tx_dict or tx > node_tx_dict[node_id]:
                    node_tx_dict[node_id] = tx

        # 获取最终交易列表
        selected_txs = list(node_tx_dict.values())
        print(f"🔍 After deduplication: {len(selected_txs)} transactions from {len(node_tx_dict)} nodes")
        print(f"   Node distribution: {node_tx_dict}")

                # ==== 新增部分：按节点分类分组并选择Top-K ====
        from collections import defaultdict

        # 创建分类到交易的映射（分类由主要数字决定）
        category_to_txs = defaultdict(list)
        for tx in selected_txs:
            node_id = self.graph.nodes[tx].get('node_id', -1)
            if node_id == -1:
                continue  # 排除无效节点
            
            # 获取该节点的主要关注数字作为分类键
            primary_digits = self.node_primary_digits.get(node_id, [])
            category = tuple(sorted(primary_digits))  # 排序以确保不同顺序视为相同分类
            category_to_txs[category].append(tx)

        # 对每个分类选择重要性Top-K的交易
        final_selected_txs = []
        for category, txs in category_to_txs.items():
            # 获取这些交易的重要性分数
            txs_with_scores = []
            for tx in txs:
                # 确保重要性分数已计算
                if tx >= len(self.tx_importance_scores) or self.tx_importance_scores[tx] == 0.0:
                    score = get_model_importance_score(self.tx_params[tx])
                    self.tx_importance_scores[tx] = score
                else:
                    score = self.tx_importance_scores[tx]
                txs_with_scores.append((tx, score))
            
            # 按分数降序排序
            sorted_txs = sorted(txs_with_scores, key=lambda x: x[1], reverse=True)
            
            # 选择Top-K（使用全局TOP_K_MODELS参数）
            k = min(TOP_K_MODELS, len(sorted_txs))
            selected = [tx for tx, _ in sorted_txs[:k]]
            final_selected_txs.extend(selected)
            print(f"  Category {category}: Selected {k} transactions (max score {sorted_txs[0][1]:.1f})")

        # ==== 结束新增部分 ====


        # 获取对应参数并联邦平均
        params_list = [self.tx_params[tx] for tx in selected_txs]
        global_params = federated_average(params_list)

        # 评估全局模型
        


        # 创建全局交易
        global_tx_id = None
        if create_global_tx:
            global_model = CIFAR10CNN()
            global_model.load_state_dict(global_params)
            acc = evaluate(global_model, test_loader)
            global_tx_id = self.current_id
            self.graph.add_node(global_tx_id, node_id=GLOBAL_NODE_ID)
            self.node_positions[global_tx_id] = (global_tx_id * 2.2, -1)
            self.graph.add_edges_from([(p, global_tx_id) for p in selected_txs])

            self.tx_params.append(global_params)
            self.tx_accuracies.append(0.0)
            self.tx_global_accuracies.append(acc)
            self.tx_importance_scores.append(get_model_importance_score(global_params))
            
            self.current_id += 1
            self.last_global_tx = global_tx_id
            print(f"\n🌐 Created Global Transaction {global_tx_id} with {len(selected_txs)} refs")
            print(f"   global accuracy: {acc:.2f}%")

            return acc, global_params, global_tx_id

    def find_best_models_for_node(self, node_id, recent_n=N_REFERENCE_CANDIDATES):
        """为指定节点找到在其主要数字上表现最好的模型
        
        Args:
            node_id: 节点ID
            recent_n: 考虑的最近交易数量
            
        Returns:
            list: 按性能排序的交易ID列表，最好的在前
        """
        # 获取除创世块外的所有交易
        all_txs = [n for n in self.graph.nodes if n != 0]
        if not all_txs:
            return [0]  # 如果只有创世块，则返回创世块
            
        # 考虑最近的N个交易
        recent_n = min(recent_n, len(all_txs))
        recent_txs = sorted(all_txs)[-recent_n:]

        if self.last_global_tx > 0 and self.last_global_tx not in recent_txs:
            recent_txs.append(self.last_global_tx)
        
        # 评估这些交易在该节点主要数字上的表现
        performance = []
        for tx in recent_txs:
            # 确保参数存在
            if tx < len(self.tx_params):
                # 评估模型在该节点主要数字上的准确率
                acc = self.evaluate_model_on_primary_digits(self.tx_params[tx], node_id)
                performance.append((tx, acc))
        
        # 按准确率降序排序
        sorted_performance = sorted(performance, key=lambda x: x[1], reverse=True)
        
        # 返回排序后的交易ID列表
        return [tx for tx, _ in sorted_performance]

    def generate_transaction(self, current_round=None):
        """生成新交易的核心方法（基于主要数字表现的引用版）"""
        if not self.graph.nodes:
            return []

        current_nodes = list(self.graph.nodes)
        selected_nodes = []
        initial_model_id = 0  # 初始模型交易ID固定为0
        new_tx_count = 0  # 新产生的交易计数
        
        # 初始化当前轮次的交易列表
        if current_round is not None and current_round not in self.round_transactions:
            self.round_transactions[current_round] = []
            self.nodes_active_in_round[current_round] = set()

        # 第一步：选择需要生成交易的节点
        for node_id in range(self.num_nodes):
            if random.random() < SELECT_CLIENTS/NUM_CLIENTS:
                # 找到在该节点主要数字上表现最好的Recent_TX个模型
                # 修改后的代码
                if self.last_global_tx > 0 and current_round %3 == 0:
                    # 存在最新的全局交易，强制引用
                    parents = [self.last_global_tx]
                else:
                    # 原有逻辑：基于主要数字选择最佳模型
                    best_models = self.find_best_models_for_node(node_id, N_REFERENCE_CANDIDATES)
                    num_parents = min(Recent_TX, len(best_models))
                    parents = best_models[:num_parents]
                    # 确保父节点不为空
                    if not parents and initial_model_id in current_nodes:
                        parents = [initial_model_id]
            
                # 如果父节点列表为空且存在创世块，则引用创世块
                if not parents and initial_model_id in current_nodes:
                    parents = [initial_model_id]

                if parents:
                    selected_nodes.append((node_id, parents))

        # 第二步：处理每个被选中的节点
        new_tx_ids = []
        for node_id, parents in selected_nodes:
            # 获取父交易的模型参数
            parent_params = []
            for p in parents:
                if p < len(self.tx_params):  # 确保参数存在
                    parent_params.append(self.tx_params[p])

            new_tx_id = self.current_id
            self.latest_tx_per_node[node_id] = new_tx_id  # 更新最新交易

            # 联邦平均逻辑
            if parent_params:
                aggregated_params = federated_average(parent_params)
            else:
                aggregated_params = self.tx_params[0]

            # 初始化本地模型
            local_model = CIFAR10CNN()
            local_model.load_state_dict(aggregated_params)
            optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
            updated_params = client_update(
                model=local_model,
                loader=train_loaders[node_id],
                optimizer=optimizer
            )
            
            # 计算参数重要性分数
            importance_score = get_model_importance_score(updated_params)

            # 创建新交易
            new_tx_id = self.current_id
            self.graph.add_node(new_tx_id, node_id=node_id)  # 关键！记录生成节点
            self.node_positions[new_tx_id] = (
                new_tx_id * 2.2,
                node_id + random.uniform(-0.3, 0.3)
            )
            self.graph.add_edges_from([(p, new_tx_id) for p in parents])
            self.tx_params.append(updated_params)
            self.tx_importance_scores.append(importance_score)
            
            # 评估并记录新模型的准确率
            print(f"\n✅ tx {new_tx_id}")
            print(f"   node: N{node_id}")
            print(f"   ref: {parents}")
            print(f"   importance score: {importance_score:.2f}")
            
            # 打印该节点主要关注的数字
            primary_digits = self.node_primary_digits[node_id]
            print(f"   primary digits: {primary_digits}")
            
            # 1. 使用节点专用测试集评估（只包含其主要数字）
            print("   acc on primary digits:", end=" ")
            local_model.load_state_dict(updated_params)
            primary_acc = self.evaluate_model_accuracy(updated_params, node_id)
            self.tx_accuracies.append(primary_acc)
            
            # 2. 使用全局测试集评估（所有10个数字）

            self.current_id += 1
            new_tx_ids.append(new_tx_id)
            new_tx_count += 1
            
            # 记录当前轮次的交易
            if current_round is not None:
                self.round_transactions[current_round].append((new_tx_id, node_id, primary_acc))
                self.nodes_active_in_round[current_round].add(node_id)
        
        # 检查是否需要执行全局聚合（每产生GLOBAL_AGGREGATION_INTERVAL个新交易）
        if new_tx_count > 0 and current_round %3 == 2:
            print(f"\n🌐 Triggering Global Aggregation after {new_tx_count} new transactions")
            self.topk_sparsification_aggregation(create_global_tx=True)
            print(f"Round {current_round} global accuracy: {self.tx_global_accuracies}%")
        
        return new_tx_ids

    
    def calculate_local_average_accuracy(self):
        """计算所有节点的最新模型平均准确度"""
        total_acc = 0.0
        for node_id in range(self.num_nodes):
            tx_id = self.latest_tx_per_node[node_id]
            if tx_id >= len(self.tx_params):
                tx_id = 0  # 使用创世块参数
            model_params = self.tx_params[tx_id]
            acc = self.evaluate_model_accuracy(model_params, node_id=node_id)
            total_acc += acc
        return total_acc / self.num_nodes

    def run_simulation(self, rounds=NUM_ROUNDS):
        """运行模拟指定轮数"""
        print(f"\n🚀 Starting simulation for {rounds} rounds...")

        
        for r in range(rounds):
            current_round = r + 1
            print(f"\n===== Round {current_round}/{rounds} =====")
            
            # 每轮生成多个交易
            for _ in range(1):  # 每轮生成交易
                self.generate_transaction(current_round=current_round)
            
            # 计算当前轮次的本地平均精度
            local_avg_acc = self.calculate_local_average_accuracy()
            self.local_acc_history.append(local_avg_acc)
            
            # 在每轮结束时执行全局聚合评估（不创建全局交易）
            print(f"\n📈 Evaluating global model at round {current_round}")
            self.topk_sparsification_aggregation(create_global_tx=False)
            
            print(f"Round {current_round} local average accuracy: {local_avg_acc:.2f}%")
        
        return self.round_history, self.global_acc_history, self.local_acc_history
    
    def plot_accuracy(self):
        """绘制全局模型精度和本地平均精度随时间变化的曲线"""
        plt.figure(figsize=(12, 7))

        print("global_acc_history",self.tx_global_accuracies)
        print("local_acc_history",self.local_acc_history)
        
        # 绘制全局精度曲线
        plt.plot(self.round_history, self.global_acc_history, marker='o', 
                 linestyle='-', linewidth=2, markersize=5, label='Global Model Accuracy')
        
        # 绘制本地平均精度曲线
        plt.plot(self.round_history, self.local_acc_history, marker='s', 
                 linestyle='--', linewidth=2, markersize=5, color='orange', 
                 label='Local Average Accuracy')
        
        plt.title('Global vs Local Model Accuracy Over Rounds', fontsize=16)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(50, 100)  # 调整Y轴范围更好地展示准确率变化
        
        # 添加全局模型平均值和最大值线
        global_avg_acc = np.mean(self.global_acc_history)
        global_max_acc = np.max(self.global_acc_history)
        plt.axhline(y=global_avg_acc, color='blue', linestyle=':', alpha=0.7,
                   label=f'Global Avg: {global_avg_acc:.2f}%')
        plt.axhline(y=global_max_acc, color='g', linestyle='--', 
                    label=f'Maximum: {global_max_acc:.2f}%')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('global_accuracy.png', dpi=300)
        plt.show()

        print(self.global_acc_history)
        print(self.local_acc_history)

# 初始化系统和数据
train_loaders, client_test_loaders, test_loader, node_primary_digits = load_data_dirichlet()
global_model = CIFAR10CNN().to(DEVICE)  # 初始化全局模型
print("Initial model accuracy:", evaluate(global_model, test_loader))

# 创建并运行模拟
sim = DAGBlockchainFL(num_nodes=NUM_CLIENTS, node_primary_digits=node_primary_digits)
sim.run_simulation(rounds=NUM_ROUNDS)
sim.plot_accuracy()

