import argparse
import math
import os.path as osp
# import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import random
from torch.distributions import Normal  # 添加这行导入
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from sklearn.manifold import TSNE
from time import perf_counter as t

from torch_geometric.graphgym import optim
from torch.optim.lr_scheduler import ExponentialLR
from graphData import graphDataset
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GATConv
import os
from model import Encoder, GATModel, drop_feature
from models.model import Decoder
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from torch_geometric.transforms import Compose, NormalizeFeatures, ToDevice, RandomLinkSplit
from collections import deque
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


class ResidualBlock(nn.Module):
    """简化版残差块，包含单层线性变换的跳跃连接"""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)  # 单层线性变换

    def forward(self, x):
        return x + self.linear(x)  # 直接将输入与线性变换结果相加


class EnhancedDistancePredictor(nn.Module):
    """增强版距离预测网络"""

    def __init__(self, input_dim, hidden_dim=512, depth=1):
        super().__init__()

        # 初始降维层
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 中间处理层
        layers = []

        # 第一层残差块 - 降维到hidden_dim//2
        layers.append(nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        ))

        # 多个残差块，维度为hidden_dim//2
        for _ in range(depth):
            layers.append(ResidualBlock(hidden_dim // 2))

        # 添加注意力机制
        # layers.append(nn.Sequential(
        #     nn.LayerNorm(hidden_dim // 2),
        #     nn.Linear(hidden_dim // 2, hidden_dim // 2),
        #     nn.GELU(),
        #     nn.Dropout(0.1)
        # ))

        # 最终预测层
        layers.append(nn.Sequential(
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1)
        ))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # 初始投影
        x = self.input_proj(x)

        # 应用中间层
        x = self.layers[0](x)  # 降维层

        # 应用残差块
        for i in range(1, len(self.layers) - 1):
            x = self.layers[i](x)

        # 最终预测
        x = self.layers[-1](x)

        return x.squeeze(-1)
# === 扩散模型相关组件 ===
class GaussianDiffusion:#实现高斯扩散过程，包括前向加噪和反向去噪
    """
    实现高斯扩散过程，包括前向加噪与反向去噪
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # 线性噪声调度
        # $\beta_t = \text{linspace}(\beta_\text{start}, \beta_\text{end}, T)$
        # self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        # 计算alpha和相关参数
        self.alphas = 1. - self.betas# $\alpha_t = 1 - \beta_t$
        # $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # $\bar{\alpha}_{t-1}$, 其中 $\bar{\alpha}_0 = 1$
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        # $\sqrt{\bar{\alpha}_t}$
        # 计算扩散过程中需要的参数,bar表示累计乘积
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # $\sqrt{1 - \bar{\alpha}_t}$
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # 计算后验方差
        self.posterior_variance = self.betas * (
                    1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):#原始数据 x₀ 直接计算出时间步 t 的噪声数据 x_t，而不是从 x_t 到 x_{t+1} 的单步转换
        """添加噪声的前向过程"""
        if noise is None:
            noise = torch.randn_like(x_0)

        # 获取对应时间步的参数
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # 应用扩散公式: x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        #就是从原始干净数据 x₀ 一步到位地计算出任意时间步 t 的噪声数据 x_t。
        #噪声确实是逐步添加的,但在实际实现中，我们可以利用数学推导直接计算出任意时间步的噪声状态。
    def p_losses(self, denoise_model, x_0, t, noise=None):#denoise_model：需要训练的去噪模型
        """计算去噪模型的损失函数"""
        if noise is None:
            noise = torch.randn_like(x_0)#这也是高斯噪声

        # 添加噪声得到x_t
        x_t = self.q_sample(x_0, t, noise)#一次添加了多步的噪声，噪声合成一个了

        # 使用去噪模型预测噪声
        predicted_noise = denoise_model(x_t, t)#预测的是添加的全部噪声，预测一个合成的噪声

        # 计算简单的均方误差损失
        loss = nn.MSELoss()(predicted_noise, noise)#noise是原来的噪声，predicted_noise是预测后的噪声

        return loss, predicted_noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):#反向过程的单步去噪，从 x_t 生成 x_{t-1}
        """单步去噪采样"""
        # 获取模型参数
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        # 预测噪声
        predicted_noise = model(x_t, t)#这里输出的是噪声,这个model是DenoiseNet

        # 计算均值，理解为当前值减去预测的噪声就能得到上一步添加噪声之前的数据的中心值
        ## 3. 计算均值（去噪后的中心点）
        mean = sqrt_recip_alphas_t * (
                    x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        # 只在t>0时添加噪声,如果只取均值（即确定性去噪），生成过程会完全固定，导致所有样本收敛到相同结果
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = extract(self.posterior_variance, t, x_t.shape)
            return mean + torch.sqrt(variance) * noise#这个才是单步去噪后的数据
        else:#当 t=0 时，表示已到达生成过程的终点（即干净数据 此时不再需要噪声
            return mean

    @torch.no_grad()# 整个函数中的操作都不会计算梯度
    def denoise(self, model, x_t, t_start):#多次单步去噪，最后还原数据
        """从指定时间步开始去噪"""
        x = x_t.clone()

        # 从t_start开始逐步去噪
        for t in reversed(range(t_start + 1)):
            t_batch = torch.full((x.shape[0],), t, device=x.device,
                                 dtype=torch.long)
            x = self.p_sample(model, x, t_batch)

        return x


# 辅助函数：从tensor中提取适当形状的元素
def extract(a, t, shape):#a包含了扩散过程中不同时间步的预计算参数值 batch_size = t.shape[0]
    """
    从tensor a中提取对应时间步t的元素，并调整为适当的形状
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()).to(t.device)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

class SinusoidalPositionEmbeddings(nn.Module):#为时间步生成正弦位置编码
    """时间步的位置编码"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):#输入时间步比如5，输出这个时间步的嵌入向量
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DenoiseNet(nn.Module):#去噪网络，学习预测和去除噪声，输出是模型预测的噪声，而不是去噪后的嵌入向量
    """去噪模型"""     #预测当前时间步的噪声，而不是所有时间步到现在这个状态产生的噪声

    def __init__(self, input_dim, time_dim=128, hidden_dim=512):
        super().__init__()
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 主网络结构
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # 时间嵌入与特征融合
        self.final = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, input_dim),
        )

    def forward(self, x, timestep):
        # 确保时间步是张量
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=x.device)
        print(f"输入 x 的维度: {x.shape}")
        # 扩展时间步以匹配批次大小
        if timestep.shape[0] != x.shape[0]:
            timestep = timestep.expand(x.shape[0])

        # 获取时间嵌入
        t_emb = self.time_mlp(timestep)

        # 特征处理
        h = self.net(x)
        print(f"特征处理后 h 的维度: {h.shape}")
        # 特征与时间嵌入融合
        h = torch.cat([h, t_emb], dim=1)
        print(f"融合后特征 h 的维度: {h.shape}")
        result = self.final(h)
        print(f"最终输出结果的维度: {result.shape}")
        # 最终预测
        return result#输出的是噪声


class DiffusionDistancePredictor(nn.Module):#完整模型，结合扩散过程和距离预测
    """使用扩散模型预测距离的完整模型"""

    def __init__(self, input_dim, diffusion_steps=100, beta_start=1e-4,
                 beta_end=0.02, hidden_dim=512, depth=4):
        super().__init__()

        # 扩散过程
        self.diffusion = GaussianDiffusion(num_timesteps=diffusion_steps,
                                           beta_start=beta_start,
                                           beta_end=beta_end)

        # 去噪网络
        self.denoise_net = DenoiseNet(input_dim=input_dim,#这个是单步去噪
                                      hidden_dim=hidden_dim)

        # 距离预测网络
        # 使用增强版距离预测网络替换原来的简单网络
        self.distance_predictor = EnhancedDistancePredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth
        )

    def forward_diffusion(self, x_0, t):
        """前向扩散：添加噪声"""
        return self.diffusion.q_sample(x_0, t)

    def denoise(self, x_t, t_start):#这个是完全去噪后的原始数据
        """从t_start步开始去噪"""   #denoise多步去噪
        return self.diffusion.denoise(self.denoise_net, x_t, t_start)

    def predict_distance(self, x):
        """预测距离"""
        return self.distance_predictor(x).squeeze(-1)

    def get_loss(self, x_0, t):#p_losses到时间步t多步去噪的损失,在ploss里面一下添加了多步噪声，然后去噪
        """计算扩散模型损失"""
        return self.diffusion.p_losses(self.denoise_net, x_0, t)#这是添加噪声和预测噪声的损失


class GraphAntigenEnv:
    """图抗原环境，用于批处理数据"""

    def __init__(self, data, train_indices, batch_size):
        self.data = data
        self.train_indices = train_indices
        self.batch_size = batch_size
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0
        batch, batch_indices = self._get_batch()
        return batch, batch_indices

    def step(self):
        # 更新索引
        self.current_idx += self.batch_size
        done = (self.current_idx >= len(self.train_indices))

        next_batch, next_indices = self._get_batch() if not done else (
        None, None)
        return next_batch, next_indices, done

    def _get_batch(self):
        if self.current_idx + self.batch_size > len(self.train_indices):
            return None, None

        batch_indices = self.train_indices[
                        self.current_idx: self.current_idx + self.batch_size]
        src, dst = self.data.edge_index[:, batch_indices]
        batch = torch.cat([self.data.x[src], self.data.x[dst]], dim=1)
        return batch, batch_indices


def evaluate(model, data, test_indices, device, diffusion_steps=50):
    """评估模型性能"""
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for i in range(0, len(test_indices), batch_size):
            batch_indices = test_indices[i: i + batch_size]
            if len(batch_indices) == 0:
                continue

            src, dst = data.edge_index[:, batch_indices]
            batch = torch.cat([data.x[src], data.x[dst]], dim=1).to(device)

            # 添加50步噪声
            t = torch.full((batch.shape[0],), diffusion_steps, device=device,
                           dtype=torch.long)
            noisy_batch = model.forward_diffusion(batch, t)#添加50步噪声后的数据

            # 去噪
            denoised_batch = model.denoise(noisy_batch, diffusion_steps)#多步去噪后的原始数据

            # 预测距离
            pred = model.predict_distance(denoised_batch)
            predictions.append(pred)

            true_distances = data.edge_attr[batch_indices].to(device)
            true_labels.append(true_distances)

    if not predictions or not true_labels:
        return float('inf')  # 如果没有预测，返回一个大值

    predictions = torch.cat(predictions, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    mse = nn.MSELoss()(predictions, true_labels).item()
    model.train()
    return mse


# === 使用示例 ===
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gData = graphDataset("nature566H1N1")
    data = gData.data.to(device)
    num_edges = data.edge_index.shape[1]
    indices = torch.randperm(num_edges, device=device)
    train_ratio = 0.8
    split = int(train_ratio * num_edges)
    train_idx = indices[:split]
    test_idx = indices[split:]

    input_dim = 56400 * 2  # 根据原代码设定
    diffusion_steps = 100  # 扩散步数
    residual_depth = 4  # 残差块的数量
    # 初始化扩散距离预测模型
    model = DiffusionDistancePredictor(
        input_dim=input_dim,
        diffusion_steps=diffusion_steps,
        hidden_dim=512,
        depth = residual_depth
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    batch_size = 256
    train_env = GraphAntigenEnv(data, train_idx, batch_size)

    # 训练过程
    for epoch in range(1000):
        batch, batch_indices = train_env.reset()
        batch_losses = []
        denoise_losses = []
        prediction_losses = []

        while batch is not None:
            batch = batch.to(device)

            # 第一阶段：训练扩散去噪模型
            # 随机选择时间步
            t = torch.randint(0, diffusion_steps, (batch.shape[0],),
                              device=device)

            # 计算去噪损失
            denoise_loss, _ = model.get_loss(batch, t)#这是预测的噪声和真实添加的噪声之间的损失


            loss = denoise_loss
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            batch_losses.append(loss.item())
            denoise_losses.append(denoise_loss.item())
            # prediction_losses.append(pred_loss.item())

            # 进入下一批次
            batch, batch_indices, done = train_env.step()
            if done:
                break

        # 打印训练信息
        if batch_losses:
            print(f"Epoch {epoch}")
            print(f"  Total Loss: {np.mean(batch_losses):.4f}")
            print(f"  Denoise Loss: {np.mean(denoise_losses):.4f}")
            print(f"  Prediction Loss: {np.mean(prediction_losses):.4f}")
        else:
            print(f"Epoch {epoch}, No valid batches found")
        scheduler.step()
        # 评估
        test_mse = evaluate(model, data, test_idx, device,
                            diffusion_steps=diffusion_steps // 2)
        print(f"  Test MSE: {test_mse:.4f}")
