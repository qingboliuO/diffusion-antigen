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
from rbsANDdecompression import rbs, DecompressionNetwork

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_parameters():
    """打印各个模型组件的参数量"""

    # 1. DecompressionNetwork参数量
    net = DecompressionNetwork(compressed_dim=128, original_dim=56400)
    net_params = count_parameters(net)
    print(f"DecompressionNetwork parameters: {net_params:,}")

    # 2. GATModel参数量
    node_dim = 128  # 从代码中的node_dim
    num_hidden = 128
    num_proj_hidden = 64
    encoder = Encoder(node_dim, num_hidden, F.relu, base_model=GATConv, k=2)
    encoder_model = GATModel(encoder, num_hidden, num_proj_hidden, node_dim,
                             0.3)
    encoder_params = count_parameters(encoder_model)
    print(f"GATModel parameters: {encoder_params:,}")

    # 3. NodeDiffusionModel参数量（详细分解）
    diffusion_model = NodeDiffusionModel(
        node_dim=node_dim,
        diffusion_steps=100,
        beta_start=1e-6,
        beta_end=0.001,
        hidden_dim=512
    )

    # 分别计算NodeDiffusionModel内部组件
    denoise_params = count_parameters(diffusion_model.denoise_net)
    distance_params = count_parameters(diffusion_model.distance_predictor)
    diffusion_total = count_parameters(diffusion_model)

    print(f"NodeDiffusionModel components:")
    print(f"  - Denoising Network: {denoise_params:,}")
    print(f"  - Distance Predictor: {distance_params:,}")
    print(f"  - Total NodeDiffusionModel: {diffusion_total:,}")

    # 4. 总参数量
    total_params = net_params + encoder_params + diffusion_total
    print(f"\nTotal model parameters: {total_params:,}")
    print(f"Total model parameters (M): {total_params / 1e6:.2f}M")

    return {
        'decompression': net_params,
        'encoder': encoder_params,
        'denoise_net': denoise_params,
        'distance_predictor': distance_params,
        'total': total_params
    }
def encoder_train(model: GATModel, x, edge_index):
    model.train()
    encoder_optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=0.5)[0]
    edge_index_2 = dropout_adj(edge_index, p=0.5)[0]
    x_1 = drop_feature(x, 0.1)
    x_2 = drop_feature(x, 0.15)
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


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(
            DyT(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return x + self.layer(x)


class EnhancedDistancePredictor(nn.Module):
    def __init__(self, node_dim, hidden_dim=256, depth=1):
        super().__init__()
        input_dim = node_dim * 2

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        layers = []
        layers.append(nn.Sequential(
            DyT(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5)
        ))

        for _ in range(depth):
            layers.append(ResidualBlock(hidden_dim))

        layers.append(nn.Sequential(
            DyT(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)
        ))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.layers[0](x)
        for i in range(1, len(self.layers) - 1):
            x = self.layers[i](x)
        x = self.layers[-1](x)
        return x.squeeze(-1)


class GaussianDiffusion:
    def __init__(self, num_timesteps=100, beta_start=1e-5, beta_end=0.001):
        self.num_timesteps = num_timesteps

        t = torch.arange(0, num_timesteps)
        betas = beta_start + 0.5 * (beta_end - beta_start) * (
                1 - torch.cos(t / num_timesteps * math.pi))
        self.betas = betas

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
        self.loss_weights = (self.alphas_cumprod / (
                    1 - self.alphas_cumprod)) ** 0.5

    def q_sample(self, x_0, t, noise=None):#添加噪声
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_0, t, noise=None,
                 original_sequences=None, mapper=None, rbs_weight=0.1):
        if noise is None:
            noise = torch.randn_like(x_0)

        # 前向扩散得到 x_t
        x_t = self.q_sample(x_0, t, noise)

        # 预测噪声
        predicted_noise = denoise_model(x_t, t)

        # 计算原始的扩散损失
        weights = extract(self.loss_weights, t, x_0.shape)
        diffusion_loss = torch.mean(weights * (predicted_noise - noise) ** 2)

        # 如果提供了序列信息和mapper，计算RBS损失
        rbs_loss = 0.0
        if original_sequences is not None and mapper is not None:
            # 使用预测的噪声计算上一时间步的嵌入向量
            # 注意：这里需要使用DDPM的逆向采样公式
            sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)
            betas_t = extract(self.betas, t, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )

            # 计算预测的 x_{t-1} 的均值
            predicted_x_prev = sqrt_recip_alphas_t * (
                    x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
            )
            UncompressFeatures = net(predicted_x_prev)
            # 将预测的嵌入向量转换回序列
            batch_sequences = mapper.batch_embedding_to_sequence(
                UncompressFeatures.cpu(), verbose=True
            )

            # 计算RBS保留率
            rbs_retention = rbs(gData.data.virus_sequences, batch_sequences)
            print(
                f"RBS保留率: {rbs_retention:.4f} ({rbs_retention * 100:.2f}%)")
            # 将RBS保留率转换为损失（1 - retention作为损失）
            rbs_loss = (1.0 - rbs_retention)*100

        # 组合损失
        total_loss = diffusion_loss + rbs_weight * rbs_loss

        return total_loss, predicted_noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):#单步去噪
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        predicted_noise = model(x_t, t)
        mean = sqrt_recip_alphas_t * (
                x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = extract(self.posterior_variance, t, x_t.shape)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean

    @torch.no_grad()#多步去噪
    def denoise(self, model, x_t, t_start):
        x = x_t.clone()
        for t in reversed(range(t_start + 1)):
            t_batch = torch.full((x.shape[0],), t, device=x.device,
                                 dtype=torch.long)
            x = self.p_sample(model, x, t_batch)
        return x


def extract(a, t, shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()).to(t.device)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


class SinusoidalPositionEmbeddings(nn.Module):
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
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        if len(x.shape) == 2:
            B, C = x.shape
            x = x.unsqueeze(1)
            is_2d = True
        else:
            is_2d = False
            B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.dropout(x)

        if is_2d:
            x = x.squeeze(1)

        return x + residual


class PreNormResidual(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class DenoiseNet(nn.Module):
    def __init__(self, input_dim, time_dim=128, hidden_dim=256, depth=4,
                 dropout=0.1, use_attention=True):
        super().__init__()

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

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

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

        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

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

        self.skip_connection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        ) if hidden_dim != input_dim else nn.Identity()

    def forward(self, x, timestep):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=x.device)

        if timestep.shape[0] != x.shape[0]:
            timestep = timestep.expand(x.shape[0])

        t_emb = self.time_mlp(timestep)
        h = self.input_layer(x)
        h_skip = h

        for attn_block, ff_block in self.layers:
            h = attn_block(h)

        B, N = h.shape
        h = h.view(B, N, 1)
        t_emb = t_emb.view(B, -1, 1).expand_as(h)
        h_combined = torch.cat([h, t_emb], dim=1)
        h_fused = self.fusion(h_combined.view(B, -1))

        main_output = self.output_block(h_fused)
        skip_output = self.skip_connection(h_skip)

        return main_output + skip_output


class NodeDiffusionModel(nn.Module):
    def __init__(self, node_dim, diffusion_steps=100, beta_start=1e-4,
                 beta_end=0.02, hidden_dim=128, depth=2):
        super().__init__()

        self.diffusion = GaussianDiffusion(
            num_timesteps=diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end
        )

        self.denoise_net = DenoiseNet(
            input_dim=node_dim,
            hidden_dim=hidden_dim
        )

        self.distance_predictor = EnhancedDistancePredictor(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            depth=depth
        )

        self.node_dim = node_dim
        self.diffusion_steps = diffusion_steps

    def forward_diffusion(self, x_0, t):
        return self.diffusion.q_sample(x_0, t)

    def denoise(self, x_t, t_start):
        return self.diffusion.denoise(self.denoise_net, x_t, t_start)

    def predict_distance(self, src_features, dst_features):
        pair_features = torch.cat([src_features, dst_features], dim=1)
        return self.distance_predictor(pair_features)

    def get_diffusion_loss(self, x_0, t, original_sequences=None, mapper=None,
                           rbs_weight=1):
        return self.diffusion.p_losses(
            self.denoise_net, x_0, t,
            original_sequences=original_sequences,
            mapper=mapper,
            rbs_weight=rbs_weight
        )

    def get_prediction_loss(self, src_features, dst_features, true_distances):
        predictions = self.predict_distance(src_features, dst_features)
        return nn.MSELoss()(predictions, true_distances), predictions


def train_diffusion_model(model, data, device, optimizer, batch_size,
                          diffusion_steps, num_epochs, scheduler=None):
    print("===== Phase 1: Training Diffusion Model =====")
    best_loss = float('inf')

    all_nodes = torch.arange(data.x.shape[0])
    node_features = data.x.to(device)
    print(f"Training diffusion model on {len(all_nodes)} nodes")

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        # 直接使用原始顺序，不进行随机打乱
        for i in range(0, node_features.shape[0], batch_size):
            batch_features = node_features[i:i + batch_size]
            if len(batch_features) == 0:
                continue

            t = torch.randint(0, diffusion_steps, (batch_features.shape[0],),
                              device=device)
            loss, _ = model.get_diffusion_loss(
                batch_features, t,
                original_sequences=gData.data.virus_sequences,
                mapper=mapper,
                rbs_weight=2
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        if scheduler:
            scheduler.step()

        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')

        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Diffusion Loss = {avg_loss:.6f}")

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
    model.eval()
    all_features = data.x.to(device)
    denoised_features = []

    with torch.no_grad():
        for i in range(0, all_features.shape[0], batch_size):
            batch_features = all_features[i:i + batch_size]
            if len(batch_features) == 0:
                continue

            t = torch.full((batch_features.shape[0],), diffusion_steps,
                           device=device, dtype=torch.long)
            noisy_features = model.forward_diffusion(batch_features, t)
            batch_denoised = model.denoise(noisy_features, diffusion_steps)
            denoised_features.append(batch_denoised)

    denoised_features = torch.cat(denoised_features, dim=0)
    return denoised_features


def create_combined_training_data(data, train_indices, test_indices, device):
    """创建组合训练数据索引（不涉及具体特征）"""
    num_nodes = data.x.shape[0]

    original_src_nodes_train, original_dst_nodes_train = data.edge_index[:, train_indices]
    original_true_distances_train = data.edge_attr[train_indices].to(device)

    original_src_nodes_test, original_dst_nodes_test = data.edge_index[:, test_indices]
    original_true_distances_test = data.edge_attr[test_indices].to(device)

    diffusion_offset = num_nodes  # 扩散节点的索引偏移量
    combined_train_data = []

    # 原始节点之间的训练边
    for i in range(len(train_indices)):
        src = original_src_nodes_train[i].item()
        dst = original_dst_nodes_train[i].item()
        dist = original_true_distances_train[i].item()
        combined_train_data.append((src, dst, dist, "original-original"))

    # 扩散节点之间的训练边（保持相同抗原距离）
    for i in range(len(train_indices)):
        src = original_src_nodes_train[i].item() + diffusion_offset
        dst = original_dst_nodes_train[i].item() + diffusion_offset
        dist = original_true_distances_train[i].item()
        combined_train_data.append((src, dst, dist, "diffusion-diffusion"))

    # 测试数据（只包含原始节点）
    test_data = []
    for i in range(len(test_indices)):
        src = original_src_nodes_test[i].item()
        dst = original_dst_nodes_test[i].item()
        dist = original_true_distances_test[i].item()
        test_data.append((src, dst, dist, "original-original"))

    return combined_train_data, test_data, diffusion_offset


def create_combined_feature_mapping(data, denoised_features, diffusion_offset,
                                    device):
    node_to_feature = {}

    # 添加原始节点特征
    for node_idx in range(data.x.shape[0]):
        node_to_feature[node_idx] = data.x[node_idx].to(device)

    # 添加扩散节点特征
    for node_idx in range(data.x.shape[0]):
        diffusion_node_idx = node_idx + diffusion_offset
        node_to_feature[diffusion_node_idx] = denoised_features[node_idx].to(
            device)

    return node_to_feature


def evaluate_model_with_combined_data(model, combined_test_data,
                                      node_to_feature, device, batch_size=128):
    model.eval()

    if not combined_test_data:
        return float('inf')

    all_predictions = []
    all_true_distances = []

    with torch.no_grad():
        for i in range(0, len(combined_test_data), batch_size):
            batch_data = combined_test_data[i:i + batch_size]
            if len(batch_data) == 0:
                continue

            src_list, dst_list, dist_list, _ = zip(*batch_data)

            src_features = torch.stack(
                [node_to_feature[src] for src in src_list])
            dst_features = torch.stack(
                [node_to_feature[dst] for dst in dst_list])
            batch_true_distances = torch.tensor(dist_list, device=device,
                                                dtype=torch.float)

            predictions = model.predict_distance(src_features, dst_features)

            all_predictions.append(predictions)
            all_true_distances.append(batch_true_distances)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_true_distances = torch.cat(all_true_distances, dim=0)

    mse = nn.MSELoss()(all_predictions, all_true_distances).item()
    pred_np = all_predictions.cpu().numpy()
    true_np = all_true_distances.cpu().numpy()
    mae = mean_absolute_error(true_np, pred_np)
    r2 = r2_score(true_np, pred_np)

    return mae, mse, r2


def train_distance_predictor_with_combined_data(model, combined_train_data,
                                                test_data,
                                                node_to_feature, device,
                                                optimizer,
                                                batch_size, num_epochs,
                                                scheduler=None):
    print("===== Phase 2: Training Distance Predictor with Combined Data =====")
    best_train_loss = float('inf')
    best_test_loss = float('inf')

    print(f"Training distance predictor on {len(combined_train_data)} edges")
    print(f"Testing on {len(test_data)} original edges")

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        np.random.shuffle(combined_train_data)

        for i in range(0, len(combined_train_data), batch_size):
            batch_data = combined_train_data[i:i + batch_size]
            if len(batch_data) == 0:
                continue

            src_list, dst_list, dist_list, _ = zip(*batch_data)

            src_features = torch.stack(
                [node_to_feature[src] for src in src_list])
            dst_features = torch.stack(
                [node_to_feature[dst] for dst in dst_list])
            true_distances = torch.tensor(dist_list, device=device,
                                          dtype=torch.float)

            loss, _ = model.get_prediction_loss(src_features, dst_features,
                                                true_distances)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        if scheduler:
            scheduler.step()

        avg_train_loss = np.mean(epoch_losses) if epoch_losses else float('inf')

        if epoch % 5 != 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}")

        if epoch % 2 == 0:
            test_mae, test_loss, test_r2 = evaluate_model_with_combined_data(
                model=model,
                combined_test_data=test_data,
                node_to_feature=node_to_feature,
                device=device,
                batch_size=batch_size
            )

            print(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Test Loss = {test_loss:.6f}")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_mae = test_mae
                best_test_r2 = test_r2
                print(
                    f"  New best test loss: {best_test_loss:.6f}, Test MAE: {test_mae:.6f}, Test R²: {test_r2:.6f}")

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
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    mapper = SequenceReverseMapper()
    # 加载数据
    print("=== Model Parameter Analysis ===")
    print_model_parameters()
    print("=" * 40)
    gData = graphDataset("h3n2_2021_2612")
    print(f"使用数据集: h3n2_2021_2612")
    data = gData.data.to(device)
    num_edges = data.edge_index.shape[1]

    node_dim = data.x.shape[1]
    print(f"Node feature dimension: {node_dim}")
    net = DecompressionNetwork(compressed_dim=128, original_dim=56400)
    net.load_state_dict(torch.load('stage2_decompression_model.pth'))
    net = net.to(device)
    net.eval()

    print("=== Actual Model Parameters ===")



    # 图对比学习参数
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

    encoder_optimizer = torch.optim.Adam(encoder_model.parameters(),
                                         lr=encoder_learning_rate,
                                         weight_decay=weight_decay_encoder)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer,
                                                        step_size=200,
                                                        gamma=0.9)

    # 图对比学习训练
    start = t()
    prev = start
    for epoch in range(1, 800):
        encoder_loss = encoder_train(encoder_model, data.x, data.edge_index)
        now = t()
        print(
            f'(T) | Epoch={epoch:03d}, encoderContrastive_loss={encoder_loss:.4f}, '
            f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Encoder Training Completed ===")

    # 生成节点嵌入向量
    encoder_model.eval()
    with torch.no_grad():
        z1 = encoder_model(data.x, data.edge_index)

    print("Shape of node embeddings:", z1.shape)
    data.x = z1

    # 划分训练集和测试集
    node_dim = data.x.shape[1]
    indices = torch.arange(num_edges, device=device)
    split = 2612
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
        beta_end=0.001,
        hidden_dim=hidden_dim
    ).to(device)

    # 训练参数
    diffusion_epochs = 300#原来800
    predictor_epochs = 450

    print(
        "Starting enhanced training process with original and diffused nodes...")

    # 阶段1：训练扩散模型
    diffusion_optimizer = optim.Adam(model.denoise_net.parameters(), lr=2e-4,
                                     weight_decay=1e-5)
    diffusion_scheduler = CosineAnnealingLR(diffusion_optimizer,
                                            T_max=diffusion_epochs,
                                            eta_min=1e-6)

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
    denoised_features = generate_denoised_node_features(
        model=model,
        data=data,
        device=device,
        diffusion_steps=diffusion_steps // 2,
        batch_size=batch_size
    )

    # 创建组合训练数据和测试数据
    combined_train_data, test_data, diffusion_offset = create_combined_training_data(
        data=data,
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

    # 阶段2：训练距离预测器
    predictor_optimizer = optim.Adam(model.distance_predictor.parameters(),
                                     lr=1e-3, weight_decay=1e-5)
    predictor_scheduler = CosineAnnealingLR(predictor_optimizer,
                                            T_max=predictor_epochs,
                                            eta_min=1e-6)

    model = train_distance_predictor_with_combined_data(
        model=model,
        combined_train_data=combined_train_data,
        test_data=test_data,
        node_to_feature=combined_node_to_feature,
        device=device,
        optimizer=predictor_optimizer,
        batch_size=batch_size,
        num_epochs=predictor_epochs,
        scheduler=predictor_scheduler
    )

    checkpoint = torch.load("best_distance_predictor_by_train.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # 评估最终模型
    mae, mse, r2 = evaluate_model_with_combined_data(
        model=model,
        combined_test_data=test_data,
        node_to_feature=combined_node_to_feature,
        device=device,
        batch_size=batch_size
    )

    print(
        f"Final model evaluation - MAE: {mae:.6f}, MSE: {mse:.6f}, R²: {r2:.4f}")

    # 保存完整模型
    torch.save(model.state_dict(), "final_node_diffusion_model.pt")
    print("Training complete. Final model saved.")
