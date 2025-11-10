import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from graphData import graphDataset
from time import perf_counter as t
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GATConv
from model import Encoder, GATModel, drop_feature
from models.model import GCNDecoder
from VectorMappingSequence import SequenceReverseMapper


def rbs(original_sequences, reconstructed_sequences):
    """
    计算RBS保留率

    Args:
        original_sequences: 原始序列列表
        reconstructed_sequences: 重构序列列表

    Returns:
        float: RBS保留率 (0-1之间的值)
    """
    # RBS位点 (1-based索引转为0-based)
    rbs_positions = [147, 153, 171, 172, 202, 206, 209, 210, 238, 241, 242, 243]
    rbs_indices = [pos - 1 for pos in rbs_positions]

    total_rbs_preserved = 0
    total_rbs_checked = 0

    # 遍历所有序列对
    for orig_seq, recon_seq in zip(original_sequences, reconstructed_sequences):
        # 检查每个RBS位点
        for rbs_idx in rbs_indices:
            # 确保索引在序列范围内
            if rbs_idx < len(orig_seq) and rbs_idx < len(recon_seq):
                if orig_seq[rbs_idx] == recon_seq[rbs_idx]:
                    total_rbs_preserved += 1
                total_rbs_checked += 1

    # 计算总体RBS保留率
    if total_rbs_checked > 0:
        return total_rbs_preserved / total_rbs_checked
    else:
        return 0.0
def create_rbs_weight_mask():
    """创建RBS位点的权重掩码"""
    # RBS位点
    rbs_positions = [147, 153, 171, 172, 202, 206, 209, 210, 238, 241, 242, 243]
    # antibody_positions = [
    #     2, 3, 14, 15, 16, 18, 26, 28, 37, 41, 47, 50, 64, 66, 69, 70, 72, 73,
    #     78, 79, 91, 94, 95, 98, 99, 104, 108, 110, 112, 120, 121, 123, 133,
    #     137, 138, 140, 142, 144, 145, 147, 149, 151, 153, 154, 156, 158, 160,
    #     161, 171, 172, 173, 174, 175, 176, 179, 180, 181, 183, 188, 189, 202,
    #     204, 205, 206, 208, 209, 210, 212, 213, 217, 218, 219, 223, 224, 225,
    #     227, 228, 229, 230, 232, 233, 234, 235, 236, 238, 241, 242, 243, 245,
    #     247, 249, 258, 260, 262, 264, 276, 289, 291, 292, 294, 306, 339, 340,
    #     343, 347, 363, 369, 372, 377, 387, 391, 394, 401, 402, 422, 436, 477,
    #     484, 492, 495, 503, 505, 506, 509, 520
    # ]
    # 初始化权重掩码（全1）
    weight_mask = torch.ones(56400)

    # for antibody_pos in antibody_positions:
    #     start_idx = max(0, (antibody_pos - 3) * 100)
    #     end_idx = min(56400, antibody_pos * 100)
    #     weight_mask[start_idx:end_idx] = 1.5  # RBS位点权重为2.0
    # 为每个RBS位点设置权重
    for rbs_pos in rbs_positions:
        start_idx = max(0, (rbs_pos - 3) * 100)
        end_idx = min(56400, rbs_pos * 100)
        weight_mask[start_idx:end_idx] = 2.0  # RBS位点权重为2.0


    print(
        f"RBS权重掩码创建完成，影响维度: {(weight_mask > 1.0).sum().item()}/56400")
    return weight_mask


# 全局权重掩码（在训练开始前创建一次）
rbs_weight_mask = create_rbs_weight_mask()
if torch.cuda.is_available():
    rbs_weight_mask = rbs_weight_mask.cuda()


# 简单的位置注意力函数
def positionattention(x):
    """
    简单的RBS位点注意力函数
    Args:
        x: [batch_size, 56400] 输入嵌入向量
    Returns:
        weighted_x: [batch_size, 56400] 加权后的嵌入向量
    """
    return x * rbs_weight_mask.unsqueeze(0)
def encoder_train(model: GATModel, x, edge_index):
    model.train()
    encoder_optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=0.5)[0]
    edge_index_2 = dropout_adj(edge_index, p=0.5)[0]
    x_1 = drop_feature(x, 0.1)
    x_2 = drop_feature(x, 0.15)
    x_3 = positionattention(x_1)
    x_4 = positionattention(x_2)
    x_1 = x_1+x_3
    x_2 = x_2+x_4
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    Contrastive_loss = model.loss(z1, z2, batch_size=0)
    Contrastive_loss.backward()
    encoder_optimizer.step()
    encoder_scheduler.step()
    return Contrastive_loss.item()

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
class DecompressionNetwork(nn.Module):
    """解压缩网络 - 将压缩后的嵌入向量还原回原始维度"""

    def __init__(self, compressed_dim, original_dim, hidden_dims=None):
        super(DecompressionNetwork, self).__init__()

        if hidden_dims is None:
            # 设计逐步扩展的隐藏层维度
            hidden_dims = [
                compressed_dim * 2,  # 第一层扩展2倍
                compressed_dim * 4,  # 第二层扩展4倍
                compressed_dim * 8,  # 第三层扩展8倍
                # compressed_dim * 16,
                # compressed_dim * 32,
                # compressed_dim * 64,# 第四层扩展16倍
                original_dim  # 最终还原到原始维度
            ]

        layers = []
        prev_dim = compressed_dim

        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # 最后一层不使用激活函数和dropout
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))

        self.decoder = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, compressed_x):
        return self.decoder(compressed_x)


def train_decompression_network(decompression_model, compressed_data,
                                original_data,
                                epochs=1000, lr=0.001, device='cuda'):
    """训练解压缩网络"""

    decompression_model = decompression_model.to(device)
    compressed_data = compressed_data.to(device)
    original_data = original_data.to(device)

    # 优化器和调度器
    optimizer = optim.Adam(decompression_model.parameters(), lr=lr,
                           weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                     eta_min=1e-6)

    # 损失函数
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()

    # 训练历史
    train_losses = []
    mse_losses = []
    mae_losses = []

    print("=== Starting Decompression Network Training ===")
    decompression_model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 前向传播
        reconstructed = decompression_model(compressed_data)

        # 计算损失
        mse_loss = mse_criterion(reconstructed, original_data)
        mae_loss = mae_criterion(reconstructed, original_data)

        # 组合损失 (MSE为主，MAE为辅)
        total_loss = mse_loss + 2 * mae_loss

        # 反向传播
        total_loss.backward()

        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(decompression_model.parameters(),
                                       max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # 记录损失
        train_losses.append(total_loss.item())
        mse_losses.append(mse_loss.item())
        mae_losses.append(mae_loss.item())

        # 打印训练进度
        if epoch % 100 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch:4d}/{epochs}] | '
                  f'Total Loss: {total_loss.item():.6f} | '
                  f'MSE: {mse_loss.item():.6f} | '
                  f'MAE: {mae_loss.item():.6f} | '
                  f'LR: {current_lr:.2e}')

    print("=== Decompression Network Training Completed ===")

    # 评估最终性能
    decompression_model.eval()
    with torch.no_grad():
        final_reconstructed = decompression_model(compressed_data)
        final_mse = mse_criterion(final_reconstructed, original_data).item()
        final_mae = mae_criterion(final_reconstructed, original_data).item()

        # 计算相关系数
        original_flat = original_data.cpu().numpy().flatten()
        reconstructed_flat = final_reconstructed.cpu().numpy().flatten()
        correlation = np.corrcoef(original_flat, reconstructed_flat)[0, 1]
        r2 = r2_score(original_flat, reconstructed_flat)

        print(f"\n=== Final Performance Metrics ===")
        print(f"Final MSE: {final_mse:.6f}")
        print(f"Final MAE: {final_mae:.6f}")
        print(f"Correlation: {correlation:.4f}")
        print(f"R² Score: {r2:.4f}")

    return decompression_model, {
        'train_losses': train_losses,
        'mse_losses': mse_losses,
        'mae_losses': mae_losses,
        'final_mse': final_mse,
        'final_mae': final_mae,
        'correlation': correlation,
        'r2_score': r2
    }


# 主训练逻辑
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    gData = graphDataset("h3n2_2021_2612")
    print(f"使用数据集: h3n2_2021_2612")
    data = gData.data.to(device)
    num_edges = data.edge_index.shape[1]

    node_dim = data.x.shape[1]
    print(f"Node feature dimension: {node_dim}")

    # 图对比学习参数
    encoder_learning_rate = 0.0008
    weight_decay_encoder = 0.0005
    base_model = GATConv
    num_layers = 2
    tau = 0.3
    num_hidden = 128
    num_proj_hidden = 64

    # === 第一阶段: 图对比学习训练 ===
    print("=== STAGE 1: Graph Contrastive Learning ===")


    encoder = Encoder(node_dim, num_hidden, F.relu, base_model=base_model,
                      k=num_layers).to(device)
    encoder_model = GATModel(encoder, num_hidden, num_proj_hidden, node_dim,
                             tau).to(device)

    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(),
                                         lr=encoder_learning_rate,
                                         weight_decay=weight_decay_encoder)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer,
                                                        step_size=200,
                                                        gamma=0.9)
    #
    

    # 保存第一阶段模型
    # torch.save(encoder_model.state_dict(), 'stage1_encoder_model.pth')
    # torch.save(compressed_embeddings, 'compressed_embeddings.pth')

    # === 第二阶段: 解压缩网络训练 ===
    print("\n=== STAGE 2: Decompression Network Training ===")
    compressed_embeddings = torch.load('compressed_embeddings.pth')

    # 初始化解压缩网络
    compressed_dim = compressed_embeddings.shape[1]
    original_dim = data.x.shape[1]
    decompression_net = DecompressionNetwork(
        compressed_dim=compressed_dim,
        original_dim=original_dim,
        hidden_dims=None  # 使用默认的渐进式扩展
    )

    # # 训练解压缩网络
   

    # 保存第二阶段模型
    # torch.save(trained_decompression_net.state_dict(),
    #            'stage2_decompression_model.pth')
    decompression_net.load_state_dict(
        torch.load('stage2_decompression_model.pth', map_location=device))
    decompression_net.to(device)
    decompression_net.eval()
    # === 完整流程测试 ===
    print("\n=== Testing Complete Pipeline ===")
    encoder_model.load_state_dict(
        torch.load('stage1_encoder_model.pth', map_location=device))
    # 测试完整的压缩-解压缩流程
    encoder_model.eval()

    mapper = SequenceReverseMapper()

    with torch.no_grad():
        # 解压缩
        test_reconstructed = decompression_net(compressed_embeddings)
        reconstructed_data = test_reconstructed.cpu()
        # 将重构嵌入向量映射回氨基酸序列
        print("\n--- 处理重构序列 ---")
        reconstructed_sequences = mapper.batch_embedding_to_sequence(
            reconstructed_data,
            verbose=True
        )
        # RBS保留率统计
        rbs_retention = rbs(gData.data.virus_sequences, reconstructed_sequences)
        



