import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from skimage.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, r2_score
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.graphgym import optim as geom_optim
from graphData import graphDataset
from time import perf_counter as t
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GATConv
import os
from model import Encoder, GATModel, drop_feature
from models.model import Decoder
from scipy.stats import pearsonr
import seaborn as sns
def encoder_train(model: GATModel, x,
                  edge_index):  # edge_index通常是一个 [2, num_edges] 大小的矩阵
    model.train()  # 训练模式 encoder_model = GATModel=model
    encoder_optimizer.zero_grad()
    # epoch_counter += 1
    edge_index_1 = dropout_adj(edge_index, p=0.5)[0]  # 原来0.5
    edge_index_2 = dropout_adj(edge_index, p=0.5)[0]  # 原来0.6
    x_1 = drop_feature(x, 0.1)  # 定义了两种不同的丢特征概率，用于两个视图
    x_2 = drop_feature(x, 0.15)  # 566序列0.2和0.3好#jiah1n1 0.2 0.3效果好
    # 分别生成两个视图聚合后的嵌入向量
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)  # 模型的参数就是encoder的参数即聚合时的权重
    # 这里的z1z2是GCNModel的输出，没有经过projection的全连接层
    Contrastive_loss = model.loss(z1, z2,
                                  batch_size=0)  # 调用模型的损失函数计算两个视图的嵌入向量 z1 和 z2 之间的损失
    Contrastive_loss.backward()
    encoder_optimizer.step()  # 优化encoder生成嵌入向量过程中的权重
    encoder_scheduler.step()
    return Contrastive_loss.item()


# === 扩散模型相关组件 ===

class ResidualBlock(nn.Module):
    """简化版残差块，包含单层线性变换的跳跃连接"""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)  # 单层线性变换

    def forward(self, x):
        return x + self.linear(x)  # 直接将输入与线性变换结果相加


class EnhancedDistancePredictor(nn.Module):
    """增强版距离预测网络 - 接受连接的节点特征对"""

    def __init__(self, node_dim, hidden_dim=256, depth=1):
        super().__init__()
        # 计算两个节点特征连接后的维度
        input_dim = node_dim * 2
        # 初始降维层
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        # 中间处理层
        layers = []

        # 第一层残差块 - 降维到hidden_dim//2
        layers.append(nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5)
        ))

        # 多个残差块，维度为hidden_dim//2
        for _ in range(depth):
            layers.append(ResidualBlock(hidden_dim))

        # 最终预测层
        layers.append(nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)
        ))

        self.layers = nn.ModuleList(layers)
        # self.NodeFeatureSelfAttention = NodeFeatureSelfAttention(hidden_dim)
    def forward(self, x):
        # 初始投影
        x = self.input_proj(x)

        # 应用中间层
        x = self.layers[0](x)  # 降维层
        # x = self.NodeFeatureSelfAttention(x)
        # 应用残差块
        for i in range(1, len(self.layers) - 1):
            x = self.layers[i](x)

        # 最终预测
        x = self.layers[-1](x)

        return x.squeeze(-1)


class GaussianDiffusion:
    """实现高斯扩散过程，包括前向加噪和反向去噪"""

    def __init__(self, num_timesteps=100, beta_start=1e-5, beta_end=0.001):
        self.num_timesteps = num_timesteps

        # 使用更稳定的余弦调度参数
        # 使用非线性噪声调度，可以提高训练稳定性
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5,
                               num_timesteps) ** 2
        self.betas = betas

        # 计算alpha和相关参数
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (
                1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # 计算权重系数，让损失更加关注高噪声区域
        self.loss_weights = (self.alphas_cumprod / (
                1 - self.alphas_cumprod)) ** 0.5

    def q_sample(self, x_0, t, noise=None):
        """添加噪声的前向过程"""
        if noise is None:
            noise = torch.randn_like(x_0)

        # 获取对应时间步的参数
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # 应用扩散公式: x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_0, t, noise=None):
        """计算去噪模型的损失函数"""
        if noise is None:
            noise = torch.randn_like(x_0)

        # 添加噪声得到x_t
        x_t = self.q_sample(x_0, t, noise)

        # 使用去噪模型预测噪声
        predicted_noise = denoise_model(x_t, t)

        # 使用加权MSE损失，更加关注高噪声区域
        weights = extract(self.loss_weights, t, x_0.shape)
        loss = torch.mean(weights * (predicted_noise - noise) ** 2)

        return loss, predicted_noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """单步去噪采样"""
        # 获取模型参数
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        # 预测噪声
        predicted_noise = model(x_t, t)

        # 计算均值
        mean = sqrt_recip_alphas_t * (
                x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        #论文中公式4前半部分
        # 只在t>0时添加噪声
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = extract(self.posterior_variance, t, x_t.shape)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean

    @torch.no_grad()
    def denoise(self, model, x_t, t_start):
        """从指定时间步开始去噪"""
        x = x_t.clone()

        # 从t_start开始逐步去噪
        for t in reversed(range(t_start + 1)):
            t_batch = torch.full((x.shape[0],), t, device=x.device,
                                 dtype=torch.long)
            x = self.p_sample(model, x, t_batch)

        return x


# 辅助函数：从tensor中提取适当形状的元素
def extract(a, t, shape):
    """从tensor a中提取对应时间步t的元素，并调整为适当的形状"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()).to(t.device)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步的位置编码"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SelfAttention(nn.Module):
    """修改后的自注意力模块，可处理2D输入"""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # 保存原始输入以便最后添加残差连接
        residual = x

        # 应用层归一化
        x = self.norm(x)

        # 处理2D输入 [B, C] -> [B, 1, C]
        if len(x.shape) == 2:
            B, C = x.shape
            x = x.unsqueeze(1)  # 添加序列维度
            is_2d = True
        else:
            is_2d = False
            B, N, C = x.shape

        # 计算注意力，代码与原版相同
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.dropout(x)

        # 如果原始输入是2D，我们需要去掉序列维度
        if is_2d:
            x = x.squeeze(1)

        return x + residual


class PreNormResidual(nn.Module):
    """改进的残差块，使用PreNorm结构提高训练稳定性"""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),  # 使用SiLU(Swish)激活函数
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class DenoiseNet(nn.Module):
    """增强版去噪网络，添加自注意力机制和改进的残差结构"""

    def __init__(self, input_dim, time_dim=128, hidden_dim=256, depth=4,
                 dropout=0.1, use_attention=True):
        super().__init__()

        # 1. 时间步嵌入 - 使用更深的网络
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. 输入层 - 添加批归一化和dropout
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # 3. 主干网络 - 交替使用注意力和残差块
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if use_attention:
                self.layers.append(nn.ModuleList([
                    PreNormResidual(hidden_dim, dropout=dropout),
                    SelfAttention(hidden_dim, dropout=dropout)
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNormResidual(hidden_dim, dropout=dropout),
                    PreNormResidual(hidden_dim, dropout=dropout)
                ]))

        # 4. 时间嵌入与特征融合 - 使用门控机制
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

        # 5. 输出层 - 添加残差连接和多层结构
        self.output_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, input_dim),
        )

        # 直接跳跃连接
        self.skip_connection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        ) if hidden_dim != input_dim else nn.Identity()

    def forward(self, x, timestep):
        # 确保时间步是张量
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=x.device)

        # 扩展时间步以匹配批次大小
        if timestep.shape[0] != x.shape[0]:
            timestep = timestep.expand(x.shape[0])

        # 1. 获取时间嵌入
        t_emb = self.time_mlp(timestep)

        # 2. 输入层处理
        h = self.input_layer(x)

        # 保存初始表示用于最终跳跃连接
        h_skip = h

        # 3. 应用主干网络层
        for attn_block, ff_block in self.layers:
            h = attn_block(h)
            # h = ff_block(h)

        # 4. 特征与时间嵌入融合
        B, N = h.shape
        h = h.view(B, N, 1)  # [B, N] -> [B, N, 1] 用于自注意力
        t_emb = t_emb.view(B, -1, 1).expand_as(h)  # 调整时间嵌入维度
        h_combined = torch.cat([h, t_emb], dim=1)  # 在序列维度上连接
        h_fused = self.fusion(h_combined.view(B, -1))  # 展平并融合

        # 5. 输出预测的噪声 - 添加残差连接
        main_output = self.output_block(h_fused)
        skip_output = self.skip_connection(h_skip)

        # 组合主输出和跳跃连接
        return main_output + skip_output


class NodeDiffusionModel(nn.Module):
    """基于节点的扩散模型，在单个节点特征上执行扩散，然后连接节点特征来预测距离"""

    def __init__(self, node_dim, diffusion_steps=100, beta_start=1e-4,
                 beta_end=0.02, hidden_dim=128, depth=2):
        super().__init__()

        # 扩散过程
        self.diffusion = GaussianDiffusion(
            num_timesteps=diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end
        )

        # 去噪网络 - 处理单个节点特征
        self.denoise_net = DenoiseNet(
            input_dim=node_dim,
            hidden_dim=hidden_dim
        )

        # 距离预测网络 - 处理连接的节点特征对
        self.distance_predictor = EnhancedDistancePredictor(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            depth=depth
        )

        self.node_dim = node_dim
        self.diffusion_steps = diffusion_steps

    def forward_diffusion(self, x_0, t):
        """前向扩散：添加噪声到节点特征"""
        return self.diffusion.q_sample(x_0, t)

    def denoise(self, x_t, t_start):
        """从t_start步开始去噪节点特征"""
        return self.diffusion.denoise(self.denoise_net, x_t, t_start)

    def predict_distance(self, src_features, dst_features):
        """预测两个节点之间的距离，接受两个节点的特征"""
        # 连接源节点和目标节点的特征
        pair_features = torch.cat([src_features, dst_features], dim=1)
        return self.distance_predictor(pair_features)

    def get_diffusion_loss(self, x_0, t):
        """计算扩散模型损失（仅对节点特征）"""
        return self.diffusion.p_losses(self.denoise_net, x_0, t)

    def get_prediction_loss(self, src_features, dst_features, true_distances):
        """计算距离预测损失"""
        predictions = self.predict_distance(src_features, dst_features)
        return nn.MSELoss()(predictions, true_distances), predictions


def train_diffusion_model(model, data, device, optimizer,
                          batch_size, diffusion_steps, num_epochs,
                          scheduler=None):
    """训练扩散模型 - 使用所有节点数据"""
    print("===== Phase 1: Training Diffusion Model =====")
    best_loss = float('inf')

    # 获取所有节点的特征
    all_nodes = torch.arange(data.x.shape[0])
    node_features = data.x.to(device)
    print(f"Training diffusion model on {len(all_nodes)} nodes (ALL nodes)")

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        # 打乱数据
        perm = torch.randperm(node_features.shape[0], device=device)
        shuffled_features = node_features[perm]

        # 批量训练节点特征扩散
        for i in range(0, shuffled_features.shape[0], batch_size):
            batch_features = shuffled_features[i:i + batch_size]
            if len(batch_features) == 0:
                continue

            # 随机选择时间步
            t = torch.randint(0, diffusion_steps, (batch_features.shape[0],),
                              device=device)

            # 计算扩散损失
            loss, _ = model.get_diffusion_loss(batch_features, t)

            # 优化步骤
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_losses.append(loss.item())

        # 更新学习率
        if scheduler:
            scheduler.step()

        # 计算平均损失
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')

        # 每1个epoch报告一次
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Diffusion Loss = {avg_loss:.6f}")

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, "best_diffusion_model.pt")
                print(f"  New best model saved! Loss: {best_loss:.6f}")

    print(f"Diffusion model training completed. Best loss: {best_loss:.6f}")
    return model


def generate_denoised_node_features(model, data, device, diffusion_steps=50,
                                    batch_size=128):
    """为所有节点生成去噪特征"""
    model.eval()

    # 获取所有节点特征
    all_nodes = torch.arange(data.x.shape[0])
    all_features = data.x.to(device)

    # 为每个节点生成去噪特征
    denoised_features = []

    with torch.no_grad():
        for i in range(0, all_features.shape[0], batch_size):
            batch_features = all_features[i:i + batch_size]
            if len(batch_features) == 0:
                continue

            # 添加噪声
            t = torch.full((batch_features.shape[0],), diffusion_steps,
                           device=device, dtype=torch.long)
            noisy_features = model.forward_diffusion(batch_features, t)

            # 去噪
            batch_denoised = model.denoise(noisy_features, diffusion_steps)
            denoised_features.append(batch_denoised)

    # 合并所有去噪特征
    denoised_features = torch.cat(denoised_features, dim=0)

    # 创建节点索引到去噪特征的映射
    node_to_feature = {}
    for i, node_idx in enumerate(all_nodes.cpu().numpy()):
        node_to_feature[node_idx.item()] = denoised_features[i]

    return node_to_feature, denoised_features


def create_combined_training_data(data, denoised_features, train_indices,
                                  test_indices, device):
    """创建扩展的训练数据，去噪数据只用于训练不用于测试"""
    num_nodes = data.x.shape[0]

    # 获取原始训练和测试边
    original_src_nodes_train, original_dst_nodes_train = data.edge_index[:,
                                                         train_indices]
    original_true_distances_train = data.edge_attr[train_indices].to(device)

    original_src_nodes_test, original_dst_nodes_test = data.edge_index[:,
                                                       test_indices]
    original_true_distances_test = data.edge_attr[test_indices].to(device)

    # 创建扩散节点的偏移索引
    diffusion_offset = num_nodes

    # 创建组合训练数据
    combined_train_data = []

    # 1. 原始节点之间的训练边
    for i in range(len(train_indices)):
        src = original_src_nodes_train[i].item()
        dst = original_dst_nodes_train[i].item()
        dist = original_true_distances_train[i].item()
        combined_train_data.append((src, dst, dist, "original-original"))

    # 2. 扩散节点之间的训练边（与原始训练边对应）
    for i in range(len(train_indices)):
        src = original_src_nodes_train[i].item() + diffusion_offset
        dst = original_dst_nodes_train[i].item() + diffusion_offset
        dist = original_true_distances_train[i].item()  # 保持相同的抗原距离
        combined_train_data.append((src, dst, dist, "diffusion-diffusion"))

    # 创建测试数据 - 只包含原始节点
    test_data = []

    # 只使用原始节点的测试边
    for i in range(len(test_indices)):
        src = original_src_nodes_test[i].item()
        dst = original_dst_nodes_test[i].item()
        dist = original_true_distances_test[i].item()
        test_data.append((src, dst, dist, "original-original"))

    return combined_train_data, test_data, diffusion_offset


def create_combined_feature_mapping(data, denoised_features, diffusion_offset,
                                    device):
    """创建节点到特征的映射，包含原始节点和扩散节点"""
    node_to_feature = {}

    # 1. 添加原始节点特征
    for node_idx in range(data.x.shape[0]):
        node_to_feature[node_idx] = data.x[node_idx].to(device)

    # 2. 添加扩散节点特征
    for node_idx in range(data.x.shape[0]):
        diffusion_node_idx = node_idx + diffusion_offset
        node_to_feature[diffusion_node_idx] = denoised_features[node_idx].to(
            device)

    return node_to_feature


def evaluate_model_with_combined_data(model, combined_test_data,
                                      node_to_feature, device, batch_size=128,
                                      plot_correlation=False, save_path=None):
    """评估模型性能（使用组合数据）"""
    model.eval()

    if not combined_test_data:
        return float('inf')

    # 评估模型
    all_predictions = []
    all_true_distances = []

    with torch.no_grad():
        for i in range(0, len(combined_test_data), batch_size):
            batch_data = combined_test_data[i:i + batch_size]
            if len(batch_data) == 0:
                continue

            # 准备批次数据
            src_list, dst_list, dist_list, _ = zip(*batch_data)

            # 获取节点特征
            src_features = torch.stack(
                [node_to_feature[src] for src in src_list])
            dst_features = torch.stack(
                [node_to_feature[dst] for dst in dst_list])

            # 获取真实距离
            batch_true_distances = torch.tensor(dist_list, device=device,
                                                dtype=torch.float)

            # 预测距离
            predictions = model.predict_distance(src_features, dst_features)

            all_predictions.append(predictions)
            all_true_distances.append(batch_true_distances)

    # 合并所有预测和真实距离
    all_predictions = torch.cat(all_predictions, dim=0)
    all_true_distances = torch.cat(all_true_distances, dim=0)

    # 计算MSE
    mse = nn.MSELoss()(all_predictions, all_true_distances).item()
    # 转换为NumPy数组进行进一步分析
    pred_np = all_predictions.cpu().numpy()
    true_np = all_true_distances.cpu().numpy()
    mae = mean_absolute_error(true_np, pred_np)
    # mse = mean_squared_error(true_np, pred_np)
    r2 = r2_score(true_np, pred_np)
    if plot_correlation:
        predicted = pred_np
        actual = true_np
        scatter_data = np.column_stack((actual, predicted))

        # 计算均值
        mean = np.mean(scatter_data, axis=0)

        # 计算协方差矩阵
        cov = np.cov(scatter_data.T)

        # 计算椭圆参数（95%置信区间）
        lambda_, v = np.linalg.eigh(cov)
        lambda_ = np.sqrt(lambda_)
        angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))

        # 椭圆缩放系数（对应置信区间，2.0约为95%）
        scale = 2.5  # 可调整此参数控制椭圆大小

        # 创建椭圆对象
        ellipse = Ellipse(xy=mean,
                          width=lambda_[0] * 2 * scale,
                          height=lambda_[1] * 2 * scale,
                          angle=angle,
                          edgecolor='#5D6D7E',  # 浅橙色边框
                          linestyle='--',
                          linewidth=1,
                          facecolor='#83639f',  # 浅橙色填充
                          zorder=1,
                          alpha=0.6)

        # 绘制图形
        sns.set(style="white", context="paper", font_scale=1.2)
        plt.figure(figsize=(6, 6), dpi=600)

        # 先添加椭圆
        ax = plt.gca()
        ax.add_patch(ellipse)

        # 绘制散点图
        sc = plt.scatter(actual, predicted,
                         c='#c22f2f',
                         alpha=1,
                         edgecolors='none',
                         s=13,
                         zorder=5)

        # 自动确定合适的坐标轴范围
        # 计算椭圆边界
        cos_angle = np.abs(np.cos(np.radians(angle)))
        sin_angle = np.abs(np.sin(np.radians(angle)))
        ellipse_width = lambda_[0] * scale
        ellipse_height = lambda_[1] * scale
        ellipse_extent_x = ellipse_width * cos_angle + ellipse_height * sin_angle
        ellipse_extent_y = ellipse_width * sin_angle + ellipse_height * cos_angle

        # 计算椭圆的边界点
        ellipse_x_min = mean[0] - ellipse_extent_x
        ellipse_x_max = mean[0] + ellipse_extent_x
        ellipse_y_min = mean[1] - ellipse_extent_y
        ellipse_y_max = mean[1] + ellipse_extent_y

        # 计算数据点和椭圆共同的边界
        data_min = min(np.min(actual), np.min(predicted))
        data_max = max(np.max(actual), np.max(predicted))
        x_min = min(ellipse_x_min, data_min)
        x_max = max(ellipse_x_max, data_max)
        y_min = min(ellipse_y_min, data_min)
        y_max = max(ellipse_y_max, data_max)

        # 添加一点边距使图像更美观
        padding = 0.05 * max(x_max - x_min, y_max - y_min)
        auto_lims = [x_min - padding, x_max + padding]

        # 理想参考线
        plt.plot(auto_lims, auto_lims, '--',
                 color='#D55E00',  # 互补色橙色系
                 lw=1.2,
                 zorder=2,
                 label="Ideal")

        # 坐标轴美化
        plt.xlabel('Actual Values', fontsize=12, labelpad=8)
        plt.ylabel('Predicted Values', fontsize=12, labelpad=8)

        # 自动设置坐标轴范围
        plt.xlim(auto_lims)
        plt.ylim(auto_lims)

        # 网格和刻度优化
        plt.gca().set_aspect('equal')
        plt.tick_params(axis='both',
                        which='major',
                        labelsize=10,  # 更紧凑的刻度标签
                        length=3,
                        width=0.8)
        plt.grid(True,
                 linestyle=':',  # 改为点线
                 color='lightgray',
                 alpha=0.7,
                 zorder=1)

        # 自动生成合适的刻度
        # plt.locator_params(axis='both', nbins=8)  # 自动选择合适数量的刻度

        # 紧凑布局

        plt.savefig('NOH1N1scatterplot.pdf', format='pdf')
        plt.show()  # 显示图像

    return mae, mse, r2
    # return mse


def train_distance_predictor_with_combined_data(model, combined_train_data,
                                                test_data,
                                                node_to_feature, device,
                                                optimizer,
                                                batch_size, num_epochs,
                                                scheduler=None):
    """使用组合数据训练距离预测器，但仅使用原始数据进行测试评估"""
    print("===== Phase 2: Training Distance Predictor with Combined Data =====")
    best_train_loss = float('inf')
    best_test_loss = float('inf')

    print(
        f"Training distance predictor on {len(combined_train_data)} edges (including original and diffusion)")
    print(f"Testing only on {len(test_data)} original edges")

    for epoch in range(num_epochs):
        # ===== 训练阶段 =====
        model.train()
        epoch_losses = []

        # 打乱数据
        np.random.shuffle(combined_train_data)

        # 批量训练
        for i in range(0, len(combined_train_data), batch_size):
            batch_data = combined_train_data[i:i + batch_size]
            if len(batch_data) == 0:
                continue

            # 准备批次数据
            src_list, dst_list, dist_list, _ = zip(*batch_data)

            # 获取节点特征
            src_features = torch.stack(
                [node_to_feature[src] for src in src_list])
            dst_features = torch.stack(
                [node_to_feature[dst] for dst in dst_list])

            # 获取真实距离
            true_distances = torch.tensor(dist_list, device=device,
                                          dtype=torch.float)

            # 计算预测损失
            loss, _ = model.get_prediction_loss(src_features, dst_features,
                                                true_distances)

            # 优化步骤
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_losses.append(loss.item())

        # 更新学习率
        if scheduler:
            scheduler.step()

        # 计算平均训练损失
        avg_train_loss = np.mean(epoch_losses) if epoch_losses else float('inf')

        # 打印当前epoch的训练损失
        if epoch % 5 != 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}")

        # ===== 每5个epoch执行测试阶段 =====
        if epoch % 2 == 0:
            # 评估模型在测试集上的性能 - 仅使用原始数据
            test_mae, test_loss, test_r2 = evaluate_model_with_combined_data(
                model=model,
                combined_test_data=test_data,  # 只包含原始数据的测试集
                node_to_feature=node_to_feature,
                device=device,
                batch_size=batch_size
            )

            # 打印当前epoch的训练和测试损失
            print(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Test Loss = {test_loss:.6f}")
            # 只记录最佳测试损失，但不保存该模型
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_mae = test_mae  # 保存相应的MAE值
                best_test_r2 = test_r2  # 保存相应的R2值
                print(
                    f"  New best test loss: {best_test_loss:.6f}, Test MAE: {test_mae:.6f}, Test R²: {test_r2:.6f}")

        # 保存最佳模型（基于训练集性能）- 每个epoch都检查
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': test_loss if epoch % 5 == 0 else None,
            }, "best_distance_predictor_by_train.pt")
            print(
                f"  New best model by train saved! Train Loss: {best_train_loss:.6f}")

    print(f"Distance predictor training completed.")
    print(
        f"Best train loss: {best_train_loss:.6f}, Best test loss: {best_test_loss:.6f}, Best test MAE: {best_test_mae:.6f}, Best test R²: {best_test_r2:.6f}")
    return model


# 主训练逻辑
if __name__ == "__main__":
    # 设置随机种子，提高可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    # gData = graphDataset("test newnature566H3N2")
    # gData = graphDataset("newnature566H1N1")
    # gData = graphDataset("nature566H1N1")
    # gData = graphDataset("nature566H3N2")
    # gData = graphDataset("nature585BVictoria")
    # gData = graphDataset("nature585BYamagata")
    gData = graphDataset("nature566H1N1不对称")
    # gData = graphDataset("nature566H3N2不对称")
    # gData = graphDataset("nature585BVictoria不对称")
    # gData = graphDataset("nature585BYamagata不对称")
    # print(f"使用数据集:nature585BYamagata不对称")
    data = gData.data.to(device)
    num_edges = data.edge_index.shape[1]

    # 获取节点特征维度
    node_dim = data.x.shape[1]
    print(f"Node feature dimension: {node_dim}")

    # 图对比学习部分
    encoder_learning_rate = 0.0008
    weight_decay_encoder = 0.0005
    base_model = GATConv
    num_layers = 2
    tau = 0.3
    num_hidden = 128
    num_proj_hidden = 64

    # 初始化编码器
    encoder = Encoder(node_dim, num_hidden, F.relu, base_model=base_model,
                      k=num_layers).to(device)
    encoder_model = GATModel(encoder, num_hidden, num_proj_hidden, node_dim,
                             tau).to(device)

    # 定义优化器和调度器
    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(),
                                         lr=encoder_learning_rate,
                                         weight_decay=weight_decay_encoder)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer,
                                                        step_size=200,
                                                        gamma=0.9)
    encoderbest_loss = float('inf')  # 初始化最佳损失为无穷大
    encoderbest_epoch = 0  # 记录最佳损失的轮次
    # 图对比学习训练
    start = t()
    prev = start
    for epoch in range(1, 1000):
        encoder_loss = encoder_train(encoder_model, data.x, data.edge_index)
        now = t()
        print(
            f'(T) | Epoch={epoch:03d}, encoderContrastive_loss={encoder_loss:.4f}, '
            f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Encoder Training Completed ===")

    # 生成节点嵌入向量
    encoder_model.eval()
    # torch.save(encoder_model.state_dict(), 'best_encoderH3N2_model.pth')
    # encoder_model.load_state_dict(torch.load('encoderH3N2_model.pth'))
    with torch.no_grad():
        z1 = encoder_model(data.x, data.edge_index)

    print("Shape of node embeddings:", z1.shape)
    data.x = z1  # 使用生成的嵌入向量替换原始特征

    # 划分训练集和测试集
    node_dim = data.x.shape[1]
    indices = torch.randperm(num_edges, device=device)
    train_ratio = 0.8
    split = int(train_ratio * num_edges)
    train_idx = indices[:split]
    test_idx = indices[split:]

    print(
        f"Training on {len(train_idx)} edges, testing on {len(test_idx)} edges")

    # 模型参数
    diffusion_steps = 100
    hidden_dim = 512
    batch_size = 512

    # 初始化节点级别的扩散模型
    model = NodeDiffusionModel(
        node_dim=node_dim,
        diffusion_steps=diffusion_steps,
        beta_start=1e-5,
        beta_end=0.0001,
        hidden_dim=hidden_dim
    ).to(device)

    # 训练参数
    diffusion_epochs = 1000#原来1500
    predictor_epochs = 500#

    print(
        "Starting enhanced training process with original and diffused nodes...")

    # 阶段1：训练扩散模型
    diffusion_optimizer = optim.Adam(
        model.denoise_net.parameters(),
        lr=2e-4,
        weight_decay=1e-5
    )
    diffusion_scheduler = CosineAnnealingLR(diffusion_optimizer,
                                            T_max=diffusion_epochs,
                                            eta_min=1e-6)

    # 训练扩散模型 - 使用所有节点
    model = train_diffusion_model(
        model=model,
        data=data,
        device=device,
        optimizer=diffusion_optimizer,
        batch_size=batch_size,
        diffusion_steps=diffusion_steps,
        num_epochs=diffusion_epochs,
        scheduler=diffusion_scheduler
    )
    checkpoint = torch.load("best_diffusion_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    # 为所有节点生成去噪特征
    print("Generating denoised node features...")
    # data_x_backup = data.x.clone().detach()
    node_to_feature_original, denoised_features = generate_denoised_node_features(
        model=model,
        data=data,
        device=device,
        diffusion_steps=diffusion_steps // 2,
        batch_size=batch_size
    )
    # original_features_np = data_x_backup.cpu().numpy()
    # denoised_features_np = denoised_features.cpu().detach().numpy()
    #
    # # 获取病毒名称作为标签（假设data.virus_names存在）
    # virus_names = data.virus_names if hasattr(data, 'virus_names') else None
    #
    # # 应用t-SNE降维
    # print("正在执行t-SNE降维...")
    # tsne = TSNE(n_components=2, perplexity=15, method='barnes_hut',
    #             init='pca', n_iter=2000, early_exaggeration=11, random_state=42)
    #
    # # 合并特征用于同一t-SNE转换，保持原始和去噪特征在相同空间
    # combined_features = np.vstack([original_features_np, denoised_features_np])
    # combined_2d = tsne.fit_transform(combined_features)
    #
    # # 分离回原始和去噪的2D表示
    # original_2d = combined_2d[:len(original_features_np)]
    # denoised_2d = combined_2d[len(original_features_np):]
    #
    # # 使用K-means聚类
    # print("正在执行K-means聚类...")
    # num_clusters = 8  # 可以根据需要调整
    # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # clusters = kmeans.fit_predict(combined_2d)
    #
    # # 分离聚类结果
    # original_clusters = clusters[:len(original_features_np)]
    # denoised_clusters = clusters[len(original_features_np):]
    #
    # # 颜色方案 - 浅色和深色版本
    # colors = [
    #     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    # ]
    #
    # # 为每种颜色创建深色版本
    # dark_colors = [
    #     '#184e7d', '#d45f07', '#1f7a1f', '#a01d1d', '#6b4896',
    #     '#633a37', '#b25998', '#5a5a5a', '#8c8f1a', '#0f8c9c'
    # ]
    #
    # # 创建并保存t-SNE可视化
    # plt.figure(figsize=(14, 10))
    #
    # # 绘制原始特征（浅色）
    # for i in range(num_clusters):
    #     mask = original_clusters == i
    #     plt.scatter(
    #         original_2d[mask, 0], original_2d[mask, 1],
    #         c=[colors[i % len(colors)]], alpha=0.6, s=80,
    #         edgecolors='none', marker='o',
    #         label=f'Cluster {i} (Original)'
    #     )
    #
    # # 绘制去噪特征（深色）
    # for i in range(num_clusters):
    #     mask = denoised_clusters == i
    #     plt.scatter(
    #         denoised_2d[mask, 0], denoised_2d[mask, 1],
    #         c=[dark_colors[i % len(dark_colors)]], alpha=0.8, s=80,
    #         edgecolors='black', linewidths=0.5, marker='o',
    #         label=f'Cluster {i} (Denoised)'
    #     )
    #
    # # 如果有病毒名称，添加标签
    # if virus_names is not None:
    #     for i, name in enumerate(virus_names):
    #         plt.annotate(
    #             name,
    #             (denoised_2d[i, 0], denoised_2d[i, 1]),
    #             fontsize=8,
    #             alpha=0.7
    #         )
    #
    # # 创建图例（手动创建避免重复项）
    # legend_elements = []
    # for i in range(num_clusters):
    #     legend_elements.append(
    #         Patch(facecolor=colors[i % len(colors)], alpha=0.6,
    #               label=f'Cluster {i} (Original)')
    #     )
    #     legend_elements.append(
    #         Patch(facecolor=dark_colors[i % len(dark_colors)], alpha=0.8,
    #               label=f'Cluster {i} (Denoised)')
    #     )
    #
    # plt.title(
    #     't-SNE Visualization of Original (Light) vs Denoised (Dark) Features',
    #     fontsize=16)
    # plt.xlabel('t-SNE Component 1', fontsize=14)
    # plt.ylabel('t-SNE Component 2', fontsize=14)
    # plt.grid(alpha=0.3)
    #
    # # 添加图例，但放在图外以避免挡住散点
    # plt.legend(handles=legend_elements, loc='center left',
    #            bbox_to_anchor=(1, 0.5),
    #            fontsize=12)
    #
    # plt.tight_layout()
    # plt.savefig('tsne_original_vs_denoised.png', dpi=300, bbox_inches='tight')
    # print("t-SNE可视化已保存为 'tsne_original_vs_denoised.png'")
    # plt.show()

    # 创建组合训练数据和测试数据（只有原始数据）
    combined_train_data, test_data, diffusion_offset = create_combined_training_data(
        data=data,
        denoised_features=denoised_features,
        train_indices=train_idx,
        test_indices=test_idx,
        device=device
    )

    # 创建组合特征映射
    combined_node_to_feature = create_combined_feature_mapping(
        data=data,
        denoised_features=denoised_features,
        diffusion_offset=diffusion_offset,
        device=device
    )

    # 阶段2：训练距离预测器 - 使用组合数据
    predictor_optimizer = optim.Adam(
        model.distance_predictor.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )
    predictor_scheduler = CosineAnnealingLR(predictor_optimizer,
                                            T_max=predictor_epochs,
                                            eta_min=1e-6)

    # 使用组合数据训练距离预测器，但仅使用原始数据测试
    model = train_distance_predictor_with_combined_data(
        model=model,
        combined_train_data=combined_train_data,
        test_data=test_data,  # 只包含原始数据的测试集
        node_to_feature=combined_node_to_feature,
        device=device,
        optimizer=predictor_optimizer,
        batch_size=batch_size,
        num_epochs=predictor_epochs,
        scheduler=predictor_scheduler
    )
    checkpoint = torch.load("best_distance_predictor_by_train.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    # 评估最终模型 - 仅评估在原始数据测试集上的性能
    mae, mse, r2 = evaluate_model_with_combined_data(
        model=model,
        combined_test_data=test_data,  # 只包含原始数据的测试集
        node_to_feature=combined_node_to_feature,
        device=device,
        batch_size=batch_size,
        plot_correlation=True,  # 添加这个参数启用散点图绘制
        save_path="final_model_correlation.pdf"
    )

    print(
        f"Final model evaluation - MAE: {mae:.6f}, MSE: {mse:.6f}, R²: {r2:.4f}")

    # 保存完整模型
    torch.save(model.state_dict(), "final_node_diffusion_model.pt")
    print("Training complete. Final model saved.")