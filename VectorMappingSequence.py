import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F


class SequenceReverseMapper:
    """简化版氨基酸序列逆映射器 - 只支持MAE算法和批量处理"""

    def __init__(self, provect_path="data/protVec_100d_3grams.csv"):

        # 加载protVec词典
        self.provect = pd.read_csv(provect_path, delimiter='\t')


        # 建立索引到三元组的映射（只需要这一个方向）
        self.trigrams = list(self.provect['words'])
        self.idx_to_trigram = {i: trigram for i, trigram in
                               enumerate(self.trigrams)}

        # 提取所有三元组的向量表示并转换为torch tensor
        self.trigram_vecs_tensor = torch.tensor(
            self.provect.loc[:, self.provect.columns != 'words'].values,
            dtype=torch.float32
        )
        # print(f"三元组向量维度: {self.trigram_vecs_tensor.shape}")
        # print("✓ 逆映射器初始化完成")
    def find_closest_trigram_mae(self, target_vector):
        """
        使用MAE方法找到与目标向量最相似的三元组

        Args:
            target_vector: 目标100维向量 [100]

        Returns:
            trigram: 最相似的三元组字符串
        """
        if isinstance(target_vector, np.ndarray):
            target_vector = torch.tensor(target_vector, dtype=torch.float32)

        target_vector = target_vector.unsqueeze(0)  # [1, 100]

        # 计算平均绝对误差 (越小越好)
        distances = torch.mean(
            torch.abs(self.trigram_vecs_tensor - target_vector), dim=1
        )
        best_idx = torch.argmin(distances).item()

        return self.idx_to_trigram[best_idx]

    def batch_embedding_to_sequence(self, embedding_batch, verbose=True):
        """
        批量处理多个嵌入向量

        Args:
            embedding_batch: [batch_size, 56400] 或 [num_nodes, 56400]
            verbose: 是否显示处理进度

        Returns:
            sequences: 重构的氨基酸序列列表
        """
        if isinstance(embedding_batch, np.ndarray):
            embedding_batch = torch.tensor(embedding_batch, dtype=torch.float32)

        if len(embedding_batch.shape) == 1:
            # 单个向量，转换为批次
            embedding_batch = embedding_batch.unsqueeze(0)

        batch_size = embedding_batch.shape[0]

        if verbose:
            print(f"=== 批量序列重构: {batch_size} 个向量 ===")

        sequences = []

        for i in range(batch_size):
            if verbose and (i % 50 == 0 or i == batch_size - 1):
                print(f"处理进度: {i + 1}/{batch_size}")

            sequence = self.embedding_to_sequence(embedding_batch[i])
            sequences.append(sequence)

        if verbose:
            print(f"✓ 批量重构完成，共生成 {len(sequences)} 个序列")

        return sequences

    def batch_embedding_to_sequence(self, embedding_batch, verbose=True):
        """
        批量处理多个嵌入向量（优化版）

        Args:
            embedding_batch: [batch_size, 56400] 或 [num_nodes, 56400]
            verbose: 是否显示处理进度

        Returns:
            sequences: 重构的氨基酸序列列表
        """
        if isinstance(embedding_batch, np.ndarray):
            embedding_batch = torch.tensor(embedding_batch, dtype=torch.float32)

        if len(embedding_batch.shape) == 1:
            embedding_batch = embedding_batch.unsqueeze(0)

        batch_size = embedding_batch.shape[0]
        device = embedding_batch.device

        if verbose:
            print(f"=== 批量序列重构 (优化版): {batch_size} 个向量 ===")

        # 1. 重塑为 [batch_size, 564, 100]
        reshaped_batch = embedding_batch.view(batch_size, 564, 100)

        # 2. 获取三元组向量并移动到相同设备
        trigram_vecs = self.trigram_vecs_tensor.to(device)
        num_trigrams = trigram_vecs.size(0)

        # 3. 计算MAE距离 - 向量化操作 [batch_size, 564, num_trigrams]
        expanded_embeddings = reshaped_batch.unsqueeze(
            2)  # [batch_size, 564, 1, 100]
        expanded_trigrams = trigram_vecs.unsqueeze(0).unsqueeze(
            0)  # [1, 1, num_trigrams, 100]

        # 计算所有组合的MAE
        mae_distances = torch.mean(
            torch.abs(expanded_embeddings - expanded_trigrams),
            dim=-1
        )  # [batch_size, 564, num_trigrams]

        # 4. 找到每个位置最接近的三元组索引 [batch_size, 564]
        best_indices = torch.argmin(mae_distances, dim=2)

        # 5. 将索引转换为三元组字符串
        best_indices_cpu = best_indices.cpu().numpy()

        # 6. 并行拼接序列
        sequences = []
        for b in range(batch_size):
            if verbose and (b % 100 == 0 or b == batch_size - 1):
                print(f"处理进度: {b + 1}/{batch_size}")

            # 获取当前样本的所有三元组
            trigrams = [self.idx_to_trigram[idx] for idx in best_indices_cpu[b]]

            # 拼接序列
            if len(trigrams) == 0:
                sequences.append("")
            elif len(trigrams) == 1:
                sequences.append(trigrams[0])
            else:
                seq = trigrams[0]
                for trigram in trigrams[1:]:
                    seq += trigram[2] if len(trigram) >= 3 else trigram[-1]
                sequences.append(seq)

        if verbose:
            print(f"✓ 批量重构完成，共生成 {len(sequences)} 个序列")

        return sequences




def demonstrate_simplified_reverse_mapping():
    """演示简化版逆映射功能"""


    # 初始化逆映射器
    mapper = SequenceReverseMapper()

    # 加载数据
    print("\n=== 加载原始数据 ===")
    from graphData import graphDataset
    gData = graphDataset("h3n2_2021_2612")
    data = gData.data

    print(f"数据形状: {data.x.shape}")
    print(f"病毒株数量: {len(data.virus_names)}")

    # 批量重构演示（前10个病毒株）
    print(f"\n=== 批量重构演示 (前100个病毒株) ===")
    batch_embeddings = data.x[:10]
    batch_virus_names = data.virus_names[:100]

    # 执行批量重构
    batch_sequences = mapper.batch_embedding_to_sequence(batch_embeddings,
                                                         verbose=True)
    print(f"batch_sequences 类型: {type(batch_sequences)}")

    if isinstance(batch_sequences, list):
        print(f"列表长度: {len(batch_sequences)}")
        if len(batch_sequences) > 0:
            print(f"第一个序列长度: {len(batch_sequences[0])}")
            print(f"第一个序列类型: {type(batch_sequences[0])}")

    elif hasattr(batch_sequences, 'shape'):  # 如果是numpy数组或tensor
        print(f"数组形状: {batch_sequences.shape}")

    else:
        print(f"长度: {len(batch_sequences)}")
    # 显示结果
    print(f"\n=== 重构结果 ===")
    for i, (name, seq) in enumerate(zip(batch_virus_names, batch_sequences)):
        print(f"{i + 1}. {name}")
        print(f"   序列: {seq}")  # 显示完整序列
        print(f"   长度: {len(seq)}")


    return mapper


# 使用示例
if __name__ == "__main__":
    # 运行演示
    mapper = demonstrate_simplified_reverse_mapping()






