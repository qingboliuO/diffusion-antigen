import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
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
def encoder_train(model: GATModel, x, edge_index):#edge_index通常是一个 [2, num_edges] 大小的矩阵
    model.train()#训练模式 encoder_model = GATModel=model
    encoder_optimizer.zero_grad()
    # epoch_counter += 1
    edge_index_1 = dropout_adj(edge_index, p=0.5)[0]#原来0.5
    edge_index_2 = dropout_adj(edge_index, p=0.5)[0]#原来0.6
    x_1 = drop_feature(x, 0.1)  # 定义了两种不同的丢特征概率，用于两个视图
    x_2 = drop_feature(x, 0.15)  # 566序列0.2和0.3好#jiah1n1 0.2 0.3效果好
    #分别生成两个视图聚合后的嵌入向量
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)#模型的参数就是encoder的参数即聚合时的权重
    #这里的z1z2是GCNModel的输出，没有经过projection的全连接层
    Contrastive_loss = model.loss(z1, z2, batch_size=0)#调用模型的损失函数计算两个视图的嵌入向量 z1 和 z2 之间的损失
    Contrastive_loss.backward()
    encoder_optimizer.step()#优化encoder生成嵌入向量过程中的权重
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

    def __init__(self, node_dim, hidden_dim=512, depth=1):
        super().__init__()
        # 计算两个节点特征连接后的维度
        input_dim = node_dim * 2

        # 初始降维层
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        # 中间处理层
        layers = []

        # 第一层残差块 - 降维到hidden_dim//2
        layers.append(nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4)
        ))

        # 多个残差块，维度为hidden_dim//2
        for _ in range(depth):
            layers.append(ResidualBlock(hidden_dim))

        # 最终预测层
        layers.append(nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
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

    def __init__(self, node_dim, diffusion_steps=100, beta_start=1e-4,
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

    def get_diffusion_loss(self, x_0, t):
        """计算扩散模型损失（仅对节点特征）"""
        return self.diffusion.p_losses(self.denoise_net, x_0, t)

    def get_prediction_loss(self, src_features, dst_features, true_distances):
        """计算距离预测损失"""
        predictions = self.predict_distance(src_features, dst_features)
        return nn.MSELoss()(predictions, true_distances), predictions


def train_diffusion_model(model, data, train_indices, device, optimizer,
                          batch_size, diffusion_steps, num_epochs,
                          scheduler=None):
    """训练扩散模型"""
    print("===== Phase 1: Training Diffusion Model =====")
    best_loss = float('inf')

    # 获取要训练的节点集合（去重）
    src_nodes, dst_nodes = data.edge_index[:, train_indices]
    unique_nodes = torch.unique(torch.cat([src_nodes, dst_nodes]))
    node_features = data.x[unique_nodes].to(device)
    print(f"Training diffusion model on {len(unique_nodes)} unique nodes")

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

        # 每10个epoch报告一次
        if epoch % 10 == 0:
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

    return node_to_feature


def train_distance_predictor(model, data, train_indices, test_indices,
                             node_to_feature,
                             device, optimizer, batch_size, num_epochs,
                             scheduler=None):
    """训练距离预测器，并每5轮评估测试集性能"""
    print("===== Phase 2: Training Distance Predictor =====")
    best_train_loss = float('inf')
    best_test_loss = float('inf')

    # 获取训练边对应的节点
    src_nodes, dst_nodes = data.edge_index[:, train_indices]
    true_distances = data.edge_attr[train_indices].to(device)

    # 创建训练数据
    train_data = []
    for i in range(len(train_indices)):
        src = src_nodes[i].item()
        dst = dst_nodes[i].item()
        dist = true_distances[i].item()

        # 只有当源节点和目标节点都有去噪特征时才添加
        if src in node_to_feature and dst in node_to_feature:
            train_data.append((src, dst, dist))

    print(f"Training distance predictor on {len(train_data)} edges")

    for epoch in range(num_epochs):
        # ===== 训练阶段 =====
        model.train()
        epoch_losses = []

        # 打乱数据
        np.random.shuffle(train_data)

        # 批量训练
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            if len(batch_data) == 0:
                continue

            # 准备批次数据
            src_list, dst_list, dist_list = zip(*batch_data)

            # 获取去噪特征
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
        if epoch % 5 == 0:
            # 评估模型在测试集上的性能
            test_loss = evaluate_model(
                model=model,
                data=data,
                test_indices=test_indices,
                node_to_feature=node_to_feature,
                device=device,
                batch_size=batch_size
            )

            # 打印当前epoch的训练和测试损失
            print(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Test Loss = {test_loss:.6f}")

            # 保存最佳模型（基于测试集性能）
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'test_loss': test_loss,
                }, "best_distance_predictor_by_test.pt")
                print(
                    f"  New best model by test saved! Test Loss: {best_test_loss:.6f}")

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

    # 最后一个epoch执行一次测试评估（如果最后一个epoch不是5的倍数）
    if (num_epochs - 1) % 5 != 0:
        final_test_loss = evaluate_model(
            model=model,
            data=data,
            test_indices=test_indices,
            node_to_feature=node_to_feature,
            device=device,
            batch_size=batch_size
        )
        print(
            f"Final Epoch {num_epochs - 1}: Test Loss = {final_test_loss:.6f}")

    print(f"Distance predictor training completed.")
    print(
        f"Best train loss: {best_train_loss:.6f}, Best test loss: {best_test_loss:.6f}")
    return model


def evaluate_model(model, data, test_indices, node_to_feature, device, batch_size=128):
    """评估模型性能"""
    model.eval()

    # 获取测试边对应的节点
    src_nodes, dst_nodes = data.edge_index[:, test_indices]
    true_distances = data.edge_attr[test_indices].to(device)

    # 创建测试数据
    test_data = []
    for i in range(len(test_indices)):
        src = src_nodes[i].item()
        dst = dst_nodes[i].item()
        dist = true_distances[i].item()

        # 只有当源节点和目标节点都有去噪特征时才添加
        if src in node_to_feature and dst in node_to_feature:
            test_data.append((src, dst, dist))

    if not test_data:
        return float('inf')

    # 评估模型
    all_predictions = []
    all_true_distances = []

    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_data = test_data[i:i + batch_size]
            if len(batch_data) == 0:
                continue

            # 准备批次数据
            src_list, dst_list, dist_list = zip(*batch_data)

            # 获取去噪特征
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
    gData = graphDataset("nature566H3N2")
    data = gData.data.to(device)
    num_edges = data.edge_index.shape[1]

    # 获取节点特征维度
    node_dim = data.x.shape[1]
    print(f"Node feature dimension: {node_dim}")
    encoder_learning_rate = 0.0008
    weight_decay_encoder = 0.0005
    base_model = GATConv
    num_layers = 2  # 原来2
    tau = 0.3
    num_hidden = 256
    num_proj_hidden = 64
    encoder = Encoder(node_dim, num_hidden, F.relu, base_model=base_model,
                      k=num_layers).to(device)  # 2次聚合，同时随机初始化了参数
    encoder_model = GATModel(encoder, num_hidden, num_proj_hidden, node_dim,
                             tau).to(
        device)
    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(),
                                         lr=encoder_learning_rate,
                                         weight_decay=weight_decay_encoder)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer,
                                                        step_size=200,
                                                        gamma=0.9)
    start = t()
    prev = start
    # for epoch in range(1, 1000):  # 进行从1到num_epochs的循环，10月20号之前是2000
    #     encoder_loss = encoder_train(encoder_model, data.x,
    #                                  data.edge_index)  # 训练的时候好像没有用标签
    #     # 记录当前时间
    #     now = t()
    #     print(
    #         f'(T) | Epoch={epoch:03d}, encoderContrastive_loss={encoder_loss:.4f}, '  # 显示当前轮次和损失
    #         f'this epoch {now - prev:.4f}, total {now - start:.4f}')
    #     # 显示本轮耗时和总耗时。
    #     prev = now
    # print("=== Final ===")  # 经过很多个encoder优化出更好的gcnencoder编码器生成嵌入向量
    # encoder_model.eval()
    # with torch.no_grad():
    #     z1 = encoder_model(data.x, data.edge_index)  # 预训练数据集新生成的嵌入向量
    # print("Shape of z:", z1.shape)  # torch.Size([50, 128])
    # data.x = z1
    # 划分训练集和测试集
    indices = torch.randperm(num_edges, device=device)
    train_ratio = 0.8
    split = int(train_ratio * num_edges)
    train_idx = indices[:split]
    test_idx = indices[split:]

    print(
        f"Training on {len(train_idx)} edges, testing on {len(test_idx)} edges")

    # 模型参数
    diffusion_steps = 100  # 扩散步数
    hidden_dim = 512
    batch_size = 512  # 训练时批次大小

    # 初始化节点级别的扩散模型
    model = NodeDiffusionModel(
        node_dim=node_dim,
        diffusion_steps=diffusion_steps,
        beta_start=1e-5,  # 降低初始噪声，有助于稳定训练
        beta_end=0.0001,  # 降低最终噪声
        hidden_dim=hidden_dim
    ).to(device)

    # 训练参数
    diffusion_epochs = 1500  # 扩散模型训练轮数
    predictor_epochs = 300  # 距离预测器训练轮数

    print("Starting two-stage training process...")

    # 阶段1：训练扩散模型
    diffusion_optimizer = optim.Adam(
        model.denoise_net.parameters(),
        # Only include the neural network parameters
        lr=2e-4,
        weight_decay=1e-5
    )
    diffusion_scheduler = CosineAnnealingLR(diffusion_optimizer,
                                            T_max=diffusion_epochs,
                                            eta_min=1e-6)

    # 训练扩散模型
    model = train_diffusion_model(
        model=model,
        data=data,
        train_indices=train_idx,
        device=device,
        optimizer=diffusion_optimizer,
        batch_size=batch_size,
        diffusion_steps=diffusion_steps,
        num_epochs=diffusion_epochs,
        scheduler=diffusion_scheduler
    )

    # 为所有节点生成去噪特征
    print("Generating denoised node features...")
    node_to_feature = generate_denoised_node_features(
        model=model,
        data=data,
        device=device,
        diffusion_steps=diffusion_steps // 2,
        batch_size=batch_size
    )

    # 阶段2：训练距离预测器
    predictor_optimizer = optim.Adam(
        model.distance_predictor.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )
    predictor_scheduler = CosineAnnealingLR(predictor_optimizer,
                                            T_max=predictor_epochs,
                                            eta_min=1e-6)

    # 训练距离预测器
    model = train_distance_predictor(
        model=model,
        data=data,
        train_indices=train_idx,
        test_indices=test_idx,  # 新增参数
        node_to_feature=node_to_feature,
        device=device,
        optimizer=predictor_optimizer,
        batch_size=batch_size,
        num_epochs=predictor_epochs,
        scheduler=predictor_scheduler
    )

    # 评估最终模型
    final_mse = evaluate_model(
        model=model,
        data=data,
        test_indices=test_idx,
        node_to_feature=node_to_feature,
        device=device,
        batch_size=batch_size
    )

    print(f"Final model evaluation - MSE: {final_mse:.6f}")

    # 保存完整模型
    torch.save(model.state_dict(), "final_node_diffusion_model.pt")
    print("Training complete. Final model saved.")