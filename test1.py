import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.graphgym import optim as geom_optim
from graphData import graphDataset
from time import perf_counter as t


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

    def __init__(self, node_dim, hidden_dim=512, depth=1):
        super().__init__()
        # 计算两个节点特征连接后的维度
        input_dim = node_dim * 2

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


class GaussianDiffusion:
    """实现高斯扩散过程，包括前向加噪和反向去噪"""

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps

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


class DenoiseNet(nn.Module):
    """改进的去噪网络，使用残差连接和层归一化提高训练稳定性"""

    def __init__(self, input_dim, time_dim=128, hidden_dim=512):
        super().__init__()
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # 定义残差块
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ) for _ in range(4)
        ])

        # 时间嵌入与特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, input_dim),
        )

    def forward(self, x, timestep):
        # 确保时间步是张量
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=x.device)

        # 扩展时间步以匹配批次大小
        if timestep.shape[0] != x.shape[0]:
            timestep = timestep.expand(x.shape[0])

        # 获取时间嵌入
        t_emb = self.time_mlp(timestep)

        # 输入层处理
        h = self.input_layer(x)

        # 应用残差块
        for res_block in self.res_blocks:
            h_res = res_block(h)
            h = h + h_res  # 残差连接

        # 特征与时间嵌入融合
        h_combined = torch.cat([h, t_emb], dim=1)
        h_fused = self.fusion(h_combined)

        # 输出预测的噪声
        return self.output_layer(h_fused)


class NodeDiffusionModel(nn.Module):
    """基于节点的扩散模型，在单个节点特征上执行扩散，然后连接节点特征来预测距离"""

    def __init__(self, node_dim, diffusion_steps=1000, beta_start=1e-4,
                 beta_end=0.02, hidden_dim=512, depth=4):
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

    def get_loss(self, x_0, t):
        """计算扩散模型损失（仅对节点特征）"""
        return self.diffusion.p_losses(self.denoise_net, x_0, t)


def train_epoch(model, data, train_indices, device, optimizer, batch_size,
                diffusion_steps):
    """执行一个训练周期"""
    model.train()
    epoch_losses = []

    # 1. 获取要训练的节点集合（去重）
    src_nodes, dst_nodes = data.edge_index[:, train_indices]
    unique_nodes = torch.unique(torch.cat([src_nodes, dst_nodes]))
    node_features = data.x[unique_nodes].to(device)

    # 2. 打乱数据
    perm = torch.randperm(node_features.shape[0], device=device)
    shuffled_nodes = unique_nodes[perm]
    shuffled_features = node_features[perm]

    # 3. 批量训练节点特征扩散
    for i in range(0, shuffled_features.shape[0], batch_size):
        batch_features = shuffled_features[i:i + batch_size]

        # 随机选择时间步
        t = torch.randint(0, diffusion_steps, (batch_features.shape[0],),
                          device=device)

        # 计算扩散损失
        loss, _ = model.get_loss(batch_features, t)

        # 优化步骤
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_losses.append(loss.item())

    return np.mean(epoch_losses) if epoch_losses else float('inf')


def evaluate_model(model, data, test_indices, device, diffusion_steps=50,
                   batch_size=128):
    """评估模型性能"""
    model.eval()

    # 1. 收集测试中涉及的所有节点
    src_nodes, dst_nodes = data.edge_index[:, test_indices]
    all_test_nodes = torch.unique(torch.cat([src_nodes, dst_nodes]))

    # 2. 为所有测试节点执行扩散和去噪
    all_node_features = data.x[all_test_nodes].to(device)

    # 处理大型图时对节点批次处理
    denoised_features = []

    for i in range(0, all_node_features.shape[0], batch_size):
        batch_nodes = all_test_nodes[i:i + batch_size]
        batch_features = all_node_features[i:i + batch_size]

        # 添加噪声
        t = torch.full((batch_features.shape[0],), diffusion_steps,
                       device=device, dtype=torch.long)
        noisy_features = model.forward_diffusion(batch_features, t)

        # 去噪
        batch_denoised = model.denoise(noisy_features, diffusion_steps)
        denoised_features.append(batch_denoised)

    denoised_features = torch.cat(denoised_features, dim=0)

    # 创建节点索引到去噪特征的映射
    node_to_feature = {}
    for i, node_idx in enumerate(all_test_nodes.cpu().numpy()):
        node_to_feature[node_idx.item()] = denoised_features[i]

    # 3. 使用预测器评估边距离
    predictions = []
    true_labels = []

    for i in range(0, len(test_indices), batch_size):
        batch_indices = test_indices[i:i + batch_size]
        if len(batch_indices) == 0:
            continue

        batch_src = src_nodes[i:i + batch_size].cpu().numpy()
        batch_dst = dst_nodes[i:i + batch_size].cpu().numpy()

        # 收集批次中的源和目标节点特征
        src_features = torch.stack(
            [node_to_feature[src.item()] for src in batch_src])
        dst_features = torch.stack(
            [node_to_feature[dst.item()] for dst in batch_dst])

        # 预测距离
        pred = model.predict_distance(src_features, dst_features)
        predictions.append(pred)

        # 获取真实距离
        true_distances = data.edge_attr[batch_indices].to(device)
        true_labels.append(true_distances)

    if not predictions or not true_labels:
        return float('inf')  # 如果没有预测，返回一个大值

    predictions = torch.cat(predictions, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    mse = nn.MSELoss()(predictions, true_labels).item()

    return mse


# === 主训练逻辑 ===
if __name__ == "__main__":
    # 设置随机种子，提高可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    gData = graphDataset("nature566H1N1")
    data = gData.data.to(device)
    num_edges = data.edge_index.shape[1]

    # 获取节点特征维度
    node_dim = data.x.shape[1]
    print(f"Node feature dimension: {node_dim}")

    # 划分训练集和测试集
    indices = torch.randperm(num_edges, device=device)
    train_ratio = 0.8
    split = int(train_ratio * num_edges)
    train_idx = indices[:split]
    test_idx = indices[split:]

    print(
        f"Training on {len(train_idx)} edges, testing on {len(test_idx)} edges")

    # 模型参数
    diffusion_steps = 1000  # 扩散步数
    hidden_dim = 512
    batch_size = 512  # 训练时批次大小

    # 初始化节点级别的扩散模型
    model = NodeDiffusionModel(
        node_dim=node_dim,
        diffusion_steps=diffusion_steps,
        beta_start=1e-5,  # 降低初始噪声，有助于稳定训练
        beta_end=0.01,  # 降低最终噪声
        hidden_dim=hidden_dim
    ).to(device)

    # 使用AdamW优化器
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

    # 使用余弦退火学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)

    # 训练循环
    num_epochs = 1000
    best_loss = float('inf')

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = t()

        # 执行一个训练周期
        avg_loss = train_epoch(
            model=model,
            data=data,
            train_indices=train_idx,
            device=device,
            optimizer=optimizer,
            batch_size=batch_size,
            diffusion_steps=diffusion_steps
        )

        # 更新学习率
        scheduler.step()

        # 每10个epoch评估一次
        if epoch % 10 == 0:
            recon_error = evaluate_model(
                model=model,
                data=data,
                test_indices=test_idx,
                device=device,
                diffusion_steps=diffusion_steps // 2,
                batch_size=batch_size
            )

            epoch_time = t() - start_time
            print(f"Epoch {epoch} (Time: {epoch_time:.2f}s)")
            print(f"  Denoise Loss: {avg_loss:.6f}")
            print(f"  Distance Prediction Error: {recon_error:.6f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "best_node_diffusion_model.pt")
                print(f"  New best model saved! Loss: {best_loss:.6f}")