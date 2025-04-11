import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import numpy as np
import random
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的，如果这个设置为True，可能会使得运算效率降低
#     torch.backends.cudnn.benchmark = False
# setup_seed(42)
# class Encoder(torch.nn.Module):#用gcn聚合两次邻居节点，得到每个节点的嵌入向量
#     def __init__(self, in_channels: int, out_channels: int, activation,
#                  base_model=GATConv, k: int = 2, skip=True):#out_channels是编码器输出维度即num_hidden
#         super(Encoder, self).__init__()
#         self.base_model = base_model
# #在train函数里指定了，out_channels为num_hidden即256
#         assert k >= 2
#         self.k = k
#         self.skip = skip
#         head_out_dim = out_channels // 4
#         if not self.skip:
#             self.conv = [
#                 base_model(in_channels, head_out_dim, heads=4)]  # conv[0]
#             for _ in range(1, k - 1):  # 如果k=2下面这个循环一次都不会执行
#                 self.conv.append(
#                     base_model(out_channels, head_out_dim, heads=4))
#             self.conv.append(
#                 base_model(out_channels, head_out_dim, heads=4))  # conv[1]
#             self.conv = nn.ModuleList(self.conv)  # ModuleList是一个模块列表
#             self.activation = activation
#         else:
#             self.fc_skip = nn.Linear(in_channels, out_channels)
#             self.conv = [
#                 base_model(in_channels, head_out_dim, heads=4)]  # conv[0]
#             for _ in range(1, k):
#                 self.conv.append(
#                     base_model(out_channels, head_out_dim, heads=4))
#             self.conv = nn.ModuleList(self.conv)  # ModuleList是一个模块列表
#             self.activation = activation
#
#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
#         if not self.skip:
#             for i in range(self.k):
#                 x = self.activation(self.conv[i](x, edge_index))
#             return x
#         else:
#             h = self.activation(self.conv[0](x, edge_index))
#             hs = [self.fc_skip(x), h]
#             for i in range(1, self.k):
#                 u = sum(hs)
#                 hs.append(self.activation(self.conv[i](u, edge_index)))
#             return hs[-1]


# class Encoder(torch.nn.Module):#用gat聚合两次邻居节点，得到每个节点的嵌入向量
#     def __init__(self, in_channels: int, out_channels: int, activation,
#                  base_model=GATConv, k: int = 2):
#         super(Encoder, self).__init__()
#         self.base_model = base_model
# #在train函数里指定了，out_channels为num_hidden即256
#         assert k >= 2
#         self.k = k
#         head_out_dim = out_channels // 8
#         self.conv = [base_model(in_channels, head_out_dim,heads=8)]#conv[0]
#         for _ in range(1, k-1):#如果k=2下面这个循环一次都不会执行
#             self.conv.append(base_model(out_channels, head_out_dim,heads=8))
#         self.conv.append(base_model( out_channels, head_out_dim,heads=8))#conv[1]
#         self.conv = nn.ModuleList(self.conv)#ModuleList是一个模块列表，
# #self.conv是一个列表，有不同的图卷积层，这个列表被转换为 ModuleList 以确保它们能够被 PyTorch 正确管理。
#         self.fc = nn.Linear(in_channels, out_channels)
#         # self.dropout = nn.Dropout(p=0.2)
#         self.activation = activation
#
#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
#         # setup_seed(42)
#         res = self.fc(x)
#         # res = x
#         for i in range(self.k):#假如k=2          #可以试试加入dropout
#             #在第一次迭代中 (i=0)
#             #在第二次迭代中 (i=1)
#             # x = self.activation(self.conv[i](x, edge_index))
#             x = self.conv[i](x, edge_index)
#             # x = self.dropout(x)  # 在激活前应用dropout
#             x = self.activation(x)  # 应用激活函数
#             # x = self.dropout(x)  # 在激活后应用dropout
#             # x = F.relu(x + res)
#             # x = x + res
#         # return F.relu(x+res)
#         # return x
#         return (x + res)
class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):  # 默认使用GCNConv
        super(Encoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k
        # 初始化GCN层，直接使用out_channels
        self.conv = [base_model(in_channels, out_channels)]  # conv[0]
        for _ in range(1, k-1):
            self.conv.append(base_model(out_channels, out_channels))
        self.conv.append(base_model(out_channels, out_channels))  # conv[1]
        self.conv = nn.ModuleList(self.conv)

        self.fc = nn.Linear(in_channels, out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        res = self.fc(x)
        for i in range(self.k):
            x = self.conv[i](x, edge_index)
            x = self.activation(x)
        return x + res  # 输出时加上残差连接


class GATModel(torch.nn.Module):#  num_hidden: 256 num_proj_hidden: 256
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,num_features: int
                 ,tau: float = 0.5):
        super(GATModel, self).__init__()
        # self.dropout = AdaptiveFeatureDropout(num_features)
        self.encoder: Encoder = encoder
        self.tau: float = tau#一个浮点数，可能用于调节学习或正则化过程

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)#num_hidden就是encoder的输出
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
#fc1在projection使用
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        # setup_seed(42)
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:#输入是return self.encoder(x, edge_index)
        # setup_seed(42)
        z = F.elu(self.fc1(z))
        return self.fc2(z)#这里才是输入到损失函数的嵌入向量
#类型注解torch.Tensor，-> torch.Tensor说明函数的参数类型和返回类型，也可以不使用
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):#假如3个节点，则生成一个3*3的相似度矩阵
        # setup_seed(42)
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)#对输入图 z1 和 z2 进行正则化
        return torch.mm(z1, z2.t())#点积相似性：计算 视图z1 和 视图z2 的相似度矩阵

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):#一次性加载整个图
        # setup_seed(42)
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))##代表第一个视图中每个节点和另一个视图中每个节点的相似度，即负样本对的相似性
        between_sim = f(self.sim(z1, z2))#是一个正方形矩阵，
        # 只有对角线，代表正样本对的相似性
#between_sim.diag()增大（正样本）或者refl_sim（负样本）减小，都会让损失减小
        return -torch.log(#论文中公式1
            between_sim.diag()#对角线即两个不同视图中的同一个节点相似度，正样本损失
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
#refl_sim.sum(1)- refl_sim.diag()代表同一个视图中，每个节点与其它所有节点的损失，减去了自身
#between_sim.sum(1)实际上是两个部分1，代表第一个视图中每个节点与另一个视图中其它所有节点的相似度（不包括自身这个位置）2，就是between_sim.diag()即两个不同视图中的同一个节点相似度，两部分加起来等于这个公式
    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,#分批次加载图中节点
                          batch_size: int):
        # setup_seed(42)
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)#图中节点数量
        num_batches = (num_nodes - 1) // batch_size + 1#将所有节点分成几个batch
        f = lambda x: torch.exp(x / self.tau)#公式1中的那个温度系数
        indices = torch.arange(0, num_nodes).to(device)#给所有节点生成一个索引
        losses = []

        for i in range(num_batches):#第几组
            mask = indices[i * batch_size:(i + 1) * batch_size]#从全部节点的索引数组中提取当前批次的节点索引。
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N] B代表batch_size，N是所有节点
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(#结果是一个，批次内节点数量*1的矩阵，比如5*1
                #第一个维度是行没有指定代表全选，第二个维度是列指定了从i * batch_size到(i + 1) * batch_size表示选取这个矩阵中所有行这些列的对角元素
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()#第i组节点的正样本相似度
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
#refl_sim.sum(1)- refl_sim[:, i * batch_size:(i + 1) * batch_size第i组里每个节点和同一个视图中除了本身的其它所有节点相似度之和
#between_sim.sum(1)第i组里每个节点和另一个视图中其它所有节点包括自身的相似度之和，结果是一个batch_size*1的矩阵
        return torch.cat(losses)#多个批次计算得到的损失列表合并成一个连续的张量

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        # setup_seed(42)
        h1 = self.projection(z1)#图1生成的嵌入向量
        h2 = self.projection(z2)#图2生成的嵌入向量

        if batch_size == 0:#不分批次
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:#分批次加载
            l1 = self.batched_semi_loss(h1, h2, batch_size)#将图1作原图，另一个作对比
            l2 = self.batched_semi_loss(h2, h1, batch_size)#将图2作原图，另一个作对比

        ret = (l1 + l2) * 0.5#论文中公式二
        ret = ret.mean() if mean else ret.sum()#计算平均损失

        return ret

    # def drop_feature(self, x, base_drop_rate):
    #     dropped_x,weights = self.dropout(x, base_drop_rate)
    #     print("Current normalized weights:", weights)
    #     return dropped_x


def drop_feature(x, drop_prob):#接受输入张量 x 和丢弃概率 drop_prob 作为参数。
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    # drop_mask = torch.rand(
    #     (x.size(1),),
    #     dtype=torch.float32,
    #     device=x.device) < drop_prob
    #创建了一个与输入张量 x 的特征数量相同大小的空张量 drop_mask。
    #使用 torch.empty 创建一个空张量，大小为 (x.size(1), )，即特征数量。
    #将数据类型设为 torch.float32，设备为 x.device，以保持与输入张量相同的数据类型和设备。
    #使用 uniform_ 方法在范围 [0, 1) 内生成均匀分布的随机数，并与丢弃概率 drop_prob 比较，生成一个布尔掩码，表示哪些特征需要被丢弃
    x = x.clone()#对输入张量 x 进行克隆操作，以避免在原始张量上进行就地修改，保持函数的纯函数性质。
    x[:, drop_mask] = 0#使用掩码 drop_mask 将对应位置的特征值置为零，实现了特征丢弃的操作
    #表示在所有行上，将掩码为 True 的列对应的特征值置为零
    return x#返回修改后的张量 x，其中部分特征值已经被置为零，实现了特征丢弃的功能
# class AdaptiveFeatureDropout(nn.Module):
#     def __init__(self, num_features):
#         super(AdaptiveFeatureDropout, self).__init__()
#         # 初始化每个特征的权重，初始为0
#         self.feature_weights = nn.Parameter(torch.zeros(num_features))
#
#
#     def forward(self, x, base_drop_rate):
#         if self.training:
#             # 计算每个特征的调整因子，范围从0到1，使用sigmoid函数
#             weights = torch.sigmoid(self.feature_weights)
#             # 计算每个特征的丢弃概率，权重较高的特征具有更高的丢弃概率
#             drop_probs = base_drop_rate + (1 - base_drop_rate) * weights
#             # 生成丢弃mask，根据drop_probs进行伯努利采样
#             drop_mask = torch.bernoulli(1 - drop_probs).to(x.device)
#             # 应用mask
#             x = x * drop_mask
#             num_dropped_features = (drop_mask == 0).sum().item()
#             print(f"Number of dropped features: {num_dropped_features}")
#         return x
# class AdaptiveFeatureDropout(nn.Module):
#     def __init__(self, num_features):
#         super(AdaptiveFeatureDropout, self).__init__()
#         # 初始化每个特征的权重，初始权重平均分配，使得它们的和为1
#         initial_weights = torch.full((num_features,), 1 / num_features)
#         self.feature_weights = nn.Parameter(initial_weights)
#
#     def forward(self, x, base_drop_rate):
#         # 使用softmax归一化权重以确保权重之和为1
#         normalized_weights = F.softmax(self.feature_weights, dim=0)
#         # 计算每个特征的丢弃概率，这里将归一化权重乘以特征数再乘以基础丢弃率
#         drop_probs = base_drop_rate * normalized_weights * x.size(1)
#         # 生成丢弃mask
#         drop_mask = torch.bernoulli(1 - drop_probs).to(x.device)
#         # 应用mask
#         x = x * drop_mask
#         # 输出被遮蔽的特征数量
#         num_dropped_features = (drop_mask == 0).sum().item()
#         print(f"Number of dropped features: {num_dropped_features}")
#         return x

# class AdaptiveFeatureDropout(nn.Module):
#     def __init__(self, num_features):
#         super(AdaptiveFeatureDropout, self).__init__()
#         num_groups = num_features // 100  # 假设num_features能够被100整除
#         # 初始化每组特征的权重，初始权重平均分配
#         # initial_weights = torch.full((num_groups,), 1 / num_groups)
#         # initial_weights = torch.distributions.Dirichlet(
#         #     torch.ones(num_groups)).sample()
#         initial_weights = torch.distributions.Dirichlet(
#             torch.ones(num_groups)).sample() * num_groups
#         self.feature_weights = nn.Parameter(initial_weights)
#
#     def forward(self, x, base_drop_rate):
#         # 使用softmax归一化权重以确保权重之和为1
#         normalized_weights = F.softmax(self.feature_weights, dim=0)
#         # 计算每组特征的丢弃概率
#         # drop_probs = base_drop_rate * normalized_weights * (x.size(1) // 100)
#         drop_probs = base_drop_rate * normalized_weights
#         # 为每组特征生成丢弃mask
#         group_drop_mask = torch.bernoulli(1 - drop_probs).to(x.device)
#
#         # 将组mask扩展到每个特征
#         drop_mask = group_drop_mask.repeat_interleave(100)
#
#         # 应用mask
#         x = x * drop_mask
#         # 输出被遮蔽的特征数量
#         num_dropped_features = (drop_mask == 0).sum().item()
#         # print(f"Number of dropped features: {num_dropped_features}")
#         return x,normalized_weights
        # return x