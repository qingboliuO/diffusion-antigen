import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import plotly.graph_objects as go
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

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


# 使用DyT的残差块
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(
            DyT(dim),  # 使用DyT替代LayerNorm
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return x + self.layer(x)  # 残差连接


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

        # 第一层残差块 - 使用DyT替代LayerNorm
        layers.append(nn.Sequential(
            DyT(hidden_dim),  # 使用DyT替代LayerNorm
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5)
        ))

        # 多个残差块，维度为hidden_dim
        for _ in range(depth):
            layers.append(ResidualBlock(hidden_dim))

        # 最终预测层 - 使用DyT替代LayerNorm
        layers.append(nn.Sequential(
            DyT(hidden_dim),  # 使用DyT替代LayerNorm
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
    # if plot_correlation:


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
    gData = graphDataset("nature566H3N2")
    # gData = graphDataset("nature585BVictoria")
    # gData = graphDataset("nature585BYamagata")
    # gData = graphDataset("nature566H1N1不对称")
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
    for epoch in range(1, 600):
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
        beta_start=1e-6,
        beta_end=0.000001,#原来beta_end=0.0001,
        hidden_dim=hidden_dim
    ).to(device)

    # 训练参数
    diffusion_epochs = 700#原来1500
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
    data_x_backup = data.x.clone().detach()
    node_to_feature_original, denoised_features = generate_denoised_node_features(
        model=model,
        data=data,
        device=device,
        diffusion_steps=diffusion_steps // 10,
        batch_size=batch_size
    )


    original_features_np = data_x_backup.cpu().numpy()
    denoised_features_np = denoised_features.cpu().detach().numpy()

    # 执行t-SNE降维
    print("正在执行t-SNE降维...")
    tsne_original = TSNE(n_components=2, perplexity=12, method='exact',
                         init='pca', n_iter=2000, early_exaggeration=11,
                         random_state=42)
    original_2d = tsne_original.fit_transform(original_features_np)

    tsne_denoised = TSNE(n_components=2, perplexity=12, method='exact',
                         init='pca', n_iter=2000, early_exaggeration=11,
                         random_state=42)
    denoised_2d = tsne_denoised.fit_transform(denoised_features_np)

    # 只对原始特征执行K-means聚类，用于确定颜色
    print("正在执行K-means聚类...")
    num_clusters = 7
    kmeans_original = KMeans(n_clusters=num_clusters, random_state=42)
    original_clusters = kmeans_original.fit_predict(original_2d)

    # 为了计算聚类相似度，我们仍然对去噪特征进行聚类，但只用于计算相似度，不用于绘图着色
    kmeans_denoised = KMeans(n_clusters=num_clusters, random_state=42)
    denoised_clusters = kmeans_denoised.fit_predict(denoised_2d)

    # 计算聚类相似度指标
    from sklearn.metrics import adjusted_rand_score

    cluster_similarity = adjusted_rand_score(original_clusters,
                                             denoised_clusters)
    print(f"聚类相似度 (ARI): {cluster_similarity:.3f}")


    try:
        virus_names = data.virus_names  # 或者 data['virus_names'] 或者 data.name等
        print(f"✅ 成功获取 {len(virus_names)} 个毒株名称")
    except:
        print("⚠️  无法从data对象获取virus_names，使用备选方案...")

        # 方法2：如果您有其他包含毒株名称的变量
        try:
            virus_names = [f"Strain_{i}" for i in
                           range(len(original_2d))]  # 生成通用名称
            print(f"✅ 生成了 {len(virus_names)} 个通用毒株名称")
        except:
            # 方法3：基于索引生成名称
            virus_names = [f"Sample_{i + 1}" for i in range(len(original_2d))]
            print(f"✅ 基于索引生成了 {len(virus_names)} 个样本名称")


    def create_interactive_tsne_visualization(original_2d, denoised_2d,
                                              original_clusters, virus_names,
                                              num_clusters=6):
        """
        创建交互式t-SNE可视化
        """
        # 疫苗株列表
        special_viruses = [
            'A/Bangkok/1/1979', 'A/Beijing/353/1989', 'A/Brisbane/10/2007',
            'A/California/7/2004', 'A/Cambodia/e0826360/2020',
            'A/Croatia/10136RV/2023',
            'A/Darwin/6/2021', 'A/Darwin/9/2021', 'A/Fujian/411/2002',
            'A/Hong Kong/2671/2019', 'A/Hong Kong/45/2019',
            'A/Hong Kong/4801/2014',
            'A/Johannesburg/33/1994', 'A/Kansas/14/2017',
            'A/Leningrad/360/1986',
            'A/Massachusetts/18/2022', 'A/Moscow/10/99', 'A/Perth/16/2009',
            'A/Philippines/2/1982', 'A/Shangdong/9/1993', 'A/Sichuan/2/1987',
            'A/Singapore/INFIMH-16-0019/2016', 'A/South Australia/34/2019',
            'A/Switzerland/8060/2017', 'A/Switzerland/9715293/2013',
            'A/Sydney/5/1997',
            'A/Texas/50/2012', 'A/Thailand/8/2022', 'A/Victoria/361/2011',
            'A/Wellington/1/2004', 'A/Wisconsin/67/2005', 'A/Wuhan/359/1995'
        ]
        # special_viruses = [
        #     'A/Victoria/4897/2022', 'A/Wisconsin/67/2022', 'A/Sydney/5/2021',
        #     'A/Victoria/2570/2019', 'A/Wisconsin/588/2019',
        #     'A/Guangdong-Maonan/SWL1536/2019', 'A/Hawaii/70/2019',
        #     'A/Brisbane/02/2018', 'A/Michigan/45/2015', 'A/California/7/2009',
        #     'A/Brisbane/59/2007', 'A/Solomon Islands/3/2006',
        #     'A/New Caledonia/20/99', 'A/Beijing/262/95', 'A/Bayern/7/95',
        #     'A/Singapore/6/86', 'A/Chile/1/83', 'A/Brazil/11/78'
        # ]#h1n1疫苗株

        # 定义颜色方案
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # 🔧 椭圆大小调整 - 这里控制椭圆的整体大小
        # 如果要椭圆更大，同时增加两个值；要更小，同时减少两个值
        ELLIPSE_WIDTH = 2.4  # 椭圆宽度（整体大小）
        ELLIPSE_HEIGHT = 1.8  # 椭圆高度（整体大小）

        print(f"🔸 椭圆设置: 宽度={ELLIPSE_WIDTH}, 高度={ELLIPSE_HEIGHT}")
        print(f"💡 要调整椭圆大小，修改 ELLIPSE_WIDTH 和 ELLIPSE_HEIGHT 的值")
        print(f"   - 更大椭圆: 1.5, 1.2")
        print(f"   - 更小椭圆: 0.6, 0.5")
        print(f"   - 接近圆形: 1.0, 1.0")

        # 创建图形
        fig = go.Figure()

        # 添加原始特征点（圆形）
        for i in range(num_clusters):
            mask = original_clusters == i
            if np.any(mask):
                # 为该聚类的每个点创建hover文本
                cluster_virus_names = []
                cluster_vaccine_info = []
                for j in range(len(virus_names)):
                    if mask[j]:
                        virus_name = virus_names[j]
                        # 🆕 检查是否为疫苗株
                        is_vaccine = "是" if virus_name in special_viruses else "否"
                        cluster_virus_names.append(virus_name)
                        cluster_vaccine_info.append(is_vaccine)

                fig.add_trace(go.Scatter(
                    x=original_2d[mask, 0],
                    y=original_2d[mask, 1],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=colors[i % len(colors)],
                        opacity=0.7,
                        symbol='circle',
                        line=dict(width=1, color='white')
                    ),
                    text=cluster_virus_names,
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Original Features<br>' +
                                  'Cluster: ' + str(i) + '<br>' +
                                  'Vaccine Strain: %{customdata}<br>' +
                                  '<extra></extra>',
                    customdata=cluster_vaccine_info,
                    name=f'Original Cluster {i}',
                    showlegend=False
                ))

        # 🔄 修改：去噪特征使用自定义椭圆
        ellipse_shapes = []  # 用于收集椭圆shapes

        for i in range(num_clusters):
            mask = original_clusters == i
            if np.any(mask):
                # 为该聚类的每个点创建hover文本
                cluster_virus_names = []
                cluster_vaccine_info = []
                for j in range(len(virus_names)):
                    if mask[j]:
                        virus_name = virus_names[j]
                        is_vaccine = "是" if virus_name in special_viruses else "否"
                        cluster_virus_names.append(virus_name)
                        cluster_vaccine_info.append(is_vaccine)

                # 添加透明点（只用于hover交互）
                fig.add_trace(go.Scatter(
                    x=denoised_2d[mask, 0],
                    y=denoised_2d[mask, 1],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=colors[i % len(colors)],
                        opacity=0
                    ),
                    text=cluster_virus_names,
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Denoised Features<br>' +
                                  'Cluster: ' + str(i) + '<br>' +
                                  'Vaccine Strain: %{customdata}<br>' +
                                  '<extra></extra>',
                    customdata=cluster_vaccine_info,
                    name=f'Denoised Cluster {i}',
                    showlegend=False
                ))

                # 为该聚类的每个点创建椭圆
                cluster_indices = np.where(mask)[0]
                for idx in cluster_indices:
                    x_center = denoised_2d[idx, 0]
                    y_center = denoised_2d[idx, 1]

                    ellipse_shapes.append(
                        dict(
                            type="circle",
                            xref="x", yref="y",
                            x0=x_center - ELLIPSE_WIDTH / 2,  # 左边界
                            y0=y_center - ELLIPSE_HEIGHT / 2,  # 下边界
                            x1=x_center + ELLIPSE_WIDTH / 2,  # 右边界
                            y1=y_center + ELLIPSE_HEIGHT / 2,  # 上边界
                            fillcolor=colors[i % len(colors)],
                            opacity=0.9,
                            line=dict(color="black", width=1)
                        )
                    )

        print(f"✅ 创建了 {len(ellipse_shapes)} 个椭圆")

        # 🎨 简化图例 - 使用标准Plotly图例和清晰的文本标识
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=0,  # 隐藏marker
                opacity=0  # 完全透明
            ),
            name='● Original Features',
            showlegend=True
        ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=0,  # 隐藏marker
                opacity=0  # 完全透明
            ),
            name='⬬ Denoised Features',
            showlegend=True
        ))



        # 设置布局
        fig.update_layout(
            title=dict(
                text=f'H3N2 - Interactive t-SNE Visualization',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                showticklabels=False,
                showline=False,
                zeroline=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                showticklabels=False,
                showline=False,
                zeroline=False
            ),
            plot_bgcolor='white',
            width=800,
            height=800,
            shapes=ellipse_shapes,  # 包含所有椭圆（数据+图例示例）
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            hovermode='closest'
        )

        return fig


    # ========== 使用修正后的函数 ==========
    print("正在创建修正的交互式t-SNE可视化...")

    # 调用函数创建交互式图形
    interactive_fig = create_interactive_tsne_visualization(
        original_2d=original_2d,
        denoised_2d=denoised_2d,
        original_clusters=original_clusters,
        virus_names=virus_names,
        num_clusters=num_clusters
    )

    # 显示交互式图形
    interactive_fig.show()

    # 保存为HTML文件
    interactive_fig.write_html("H3N2_interactive_tsne_visualization.html")


    # ========== 🆕 疫苗株统计信息 ==========
    # special_viruses = [
    #     'A/Bangkok/1/1979', 'A/Beijing/353/1989', 'A/Brisbane/10/2007',
    #     'A/California/7/2004', 'A/Cambodia/e0826360/2020',
    #     'A/Croatia/10136RV/2023',
    #     'A/Darwin/6/2021', 'A/Darwin/9/2021', 'A/Fujian/411/2002',
    #     'A/Hong Kong/2671/2019', 'A/Hong Kong/45/2019', 'A/Hong Kong/4801/2014',
    #     'A/Johannesburg/33/1994', 'A/Kansas/14/2017', 'A/Leningrad/360/1986',
    #     'A/Massachusetts/18/2022', 'A/Moscow/10/99', 'A/Perth/16/2009',
    #     'A/Philippines/2/1982', 'A/Shangdong/9/1993', 'A/Sichuan/2/1987',
    #     'A/Singapore/INFIMH-16-0019/2016', 'A/South Australia/34/2019',
    #     'A/Switzerland/8060/2017', 'A/Switzerland/9715293/2013',
    #     'A/Sydney/5/1997',
    #     'A/Texas/50/2012', 'A/Thailand/8/2022', 'A/Victoria/361/2011',
    #     'A/Wellington/1/2004', 'A/Wisconsin/67/2005', 'A/Wuhan/359/1995'
    # ]
    #
    # vaccine_count = sum(1 for name in virus_names if name in special_viruses)
    # total_count = len(virus_names)






    # # 定义颜色方案
    # colors = [
    #     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    # ]
    #
    # # 创建椭圆标记
    # import matplotlib.path as mpath
    # import matplotlib.transforms as mtransforms
    # import numpy as np
    #
    # ellipse_path = mpath.Path.unit_circle()
    # transform = mtransforms.Affine2D().scale(1, 0.5)
    # ellipse_path = transform.transform_path(ellipse_path)
    #
    # # 创建简化的t-SNE可视化
    # plt.figure(figsize=(8, 8))
    #
    # # 先绘制数据点
    # # 绘制原始特征（圆形标记）
    # for i in range(num_clusters):
    #     mask = original_clusters == i
    #     plt.scatter(
    #         original_2d[mask, 0], original_2d[mask, 1],
    #         c=[colors[i % len(colors)]], alpha=0.7, s=100,
    #         edgecolors='white', linewidths=0.5, marker='o', zorder=2
    #     )
    #
    # # 绘制去噪特征（椭圆标记），但使用original_clusters来决定颜色
    # for i in range(num_clusters):
    #     mask = original_clusters == i  # 使用原始聚类结果来筛选点
    #     plt.scatter(
    #         denoised_2d[mask, 0], denoised_2d[mask, 1],  # 绘制去噪特征的坐标
    #         c=[colors[i % len(colors)]], alpha=0.9, s=120,  # 使用与原始特征相同的颜色
    #         edgecolors='black', linewidths=0.5, marker=ellipse_path, zorder=3
    #     )
    #
    # # 获取当前坐标轴和边界
    # ax = plt.gca()
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    #
    # # 手动绘制网格线，创建10x10的网格
    # x_ticks = np.linspace(xlim[0], xlim[1], 10)
    # y_ticks = np.linspace(ylim[0], ylim[1], 10)
    #
    # # 绘制垂直网格线（在所有图形元素之前，所以zorder=0）
    # for x in x_ticks:
    #     plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5, zorder=0)
    #
    # # 绘制水平网格线
    # for y in y_ticks:
    #     plt.axhline(y=y, color='gray', linestyle='--', alpha=0.5, zorder=0)
    #
    # # 移除刻度但保留坐标轴
    # plt.tick_params(axis='both', which='both', length=0)
    # plt.xticks([])
    # plt.yticks([])
    #
    # # 创建图例条目
    # from matplotlib.lines import Line2D
    #
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='white', label='Original Features',
    #            markerfacecolor='black', markersize=10, alpha=0.7),
    #     Line2D([0], [0], marker=ellipse_path, color='white',
    #            label='Denoised Features',
    #            markerfacecolor='black', markersize=10, alpha=0.9)
    # ]
    #
    # plt.suptitle('H1N1', fontsize=14)
    # plt.legend(handles=legend_elements, loc='upper right')
    #
    # # 保存图形
    # plt.savefig('tsne_simplified.pdf', dpi=600, bbox_inches='tight',
    #             transparent=True)
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