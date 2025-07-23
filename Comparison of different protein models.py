# 加载ESM嵌入向量
# 设置随机种子
import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from time import time as t
from sklearn.metrics import mean_absolute_error, r2_score
from graphData import graphDataset
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设置设备
import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from time import time as t
from sklearn.metrics import mean_absolute_error, r2_score
from graphData import graphDataset

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载图数据
gData = graphDataset("nature585BYamagata")
print("Node features (x) shape:", gData.data.x.shape)
print("Edge index shape:", gData.data.edge_index.shape)

data = gData.data
data.to(device)
data.edge_index = data.edge_index.to(device)
data.edge_attr = data.edge_attr.to(device)


# ========== 多模型嵌入加载器 ==========
def load_embedding_from_csv(csv_path, model_type='auto'):
    """
    加载不同类型的嵌入向量

    Args:
        csv_path: CSV文件路径
        model_type: 'esm2', 'progen2', 'biogpt', 'auto'

    Returns:
        embeddings: torch.Tensor
        model_info: dict with model information
    """
    print(f"Loading embeddings from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 自动检测嵌入类型
    embedding_prefixes = {
        'esm2': 'esm2_dim_',
        'progen2': 'progen2_dim_',
        'biogpt': 'biogpt_dim_'
    }

    detected_type = None
    embedding_cols = []

    if model_type == 'auto':
        # 自动检测
        for prefix_name, prefix in embedding_prefixes.items():
            cols = [col for col in df.columns if col.startswith(prefix)]
            if cols:
                detected_type = prefix_name
                embedding_cols = cols
                break
    else:
        # 指定类型
        if model_type in embedding_prefixes:
            prefix = embedding_prefixes[model_type]
            embedding_cols = [col for col in df.columns if
                              col.startswith(prefix)]
            detected_type = model_type

    if not embedding_cols:
        raise ValueError(f"No embedding columns found for type: {model_type}")

    # 按数字顺序排序
    embedding_cols.sort(key=lambda x: int(x.split('_')[-1]))

    # 提取模型信息
    model_info = {
        'type': detected_type,
        'embedding_dim': len(embedding_cols),
        'num_sequences': len(df)
    }

    if 'model' in df.columns:
        model_info['model_name'] = df['model'].iloc[0]
    if 'parameters' in df.columns:
        model_info['parameters'] = df['parameters'].iloc[0]
    if 'embedding_dim' in df.columns:
        model_info['reported_dim'] = df['embedding_dim'].iloc[0]

    # 提取嵌入向量
    embeddings = torch.tensor(df[embedding_cols].values, dtype=torch.float32)

    print(f"✅ Detected model type: {detected_type}")
    print(f"📊 Embedding shape: {embeddings.shape}")
    print(f"🔢 Embedding dimension: {model_info['embedding_dim']}")
    if 'model_name' in model_info:
        print(f"🤖 Model: {model_info['model_name']}")
    if 'parameters' in model_info:
        print(f"⚙️  Parameters: {model_info['parameters']}")

    return embeddings, model_info


# ========== 选择要使用的嵌入模型 ==========
print("\n=== Embedding Model Selection ===")

# 可用的嵌入文件
available_embeddings = {
    # 'esm2': 'data/nature566H3N2_sequences_ESM2_embeddings.csv',
    # 'progen2': 'data/nature566H1N1_sequences_ProGen2-base_embeddings.csv',  # 假设这是您的ProGen2文件
    # 'biogpt': 'data/nature566H3N2_sequences_BioGPT_embeddings.csv'  # 假设这是您的BioGPT文件
    # 'esm2': 'data/nature585BVictoria_sequences_ESM2_embeddings.csv',
    # 'progen2': 'data/nature585BVictoria_sequences_ProGen2-base_embeddings.csv',  # 假设这是您的ProGen2文件
    # 'biogpt': 'data/nature585BVictoria_sequences_BioGPT_embeddings.csv'  # 假设这是您的BioGPT文件
    'esm2': 'data/nature585BYamagata_sequences_ESM2_embeddings.csv',
    'progen2': 'data/nature585BYamagata_sequences_ProGen2-base_embeddings.csv',  # 假设这是您的ProGen2文件
    'biogpt': 'data/nature585BYamagata_sequences_BioGPT_embeddings.csv'
}

# 让用户选择模型或自动检测
EMBEDDING_CHOICE = 'esm2'  # 更改这里来选择不同的模型：'esm2', 'progen2', 'biogpt'

print(f"🎯 Selected embedding model: {EMBEDDING_CHOICE}")

# 根据选择加载相应的嵌入
if EMBEDDING_CHOICE in available_embeddings:
    csv_path = available_embeddings[EMBEDDING_CHOICE]
else:
    # 如果指定了具体文件路径
    csv_path = EMBEDDING_CHOICE

try:
    embeddings, model_info = load_embedding_from_csv(csv_path,
                                                     model_type='auto')
    embeddings = embeddings.to(device)

    print(f"\n📋 Model Information:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")

except FileNotFoundError:
    print(f"❌ File not found: {csv_path}")
    print("Available options:")
    for name, path in available_embeddings.items():
        print(f"   {name}: {path}")
    raise

# 验证节点数量匹配
num_nodes = data.edge_index.max().item() + 1
print(f"\n🔍 Validation:")
print(f"Graph nodes: {num_nodes}")
print(f"Embedding sequences: {embeddings.shape[0]}")

if embeddings.shape[0] != num_nodes:
    print("⚠️  Warning: Number of sequences doesn't match graph nodes")
    print("Adjusting embeddings...")

    if embeddings.shape[0] > num_nodes:
        embeddings = embeddings[:num_nodes]
        print(f"✂️  Truncated to {num_nodes} sequences")
    else:
        # 如果嵌入序列少于图节点，用零填充或重复
        diff = num_nodes - embeddings.shape[0]
        padding = torch.zeros(diff, embeddings.shape[1], device=device)
        embeddings = torch.cat([embeddings, padding], dim=0)
        print(f"📌 Padded with {diff} zero vectors")

# ========== 替换图数据的特征 ==========
data.x = embeddings
print(f"\n✅ Updated data.x shape: {data.x.shape}")

# 获取最终的嵌入维度
embedding_dim = data.x.shape[1]
print(f"📊 Final embedding dimension: {embedding_dim}")
print(f"🧬 Number of nodes: {data.x.shape[0]}")

# ========== 训练解码器预测抗原距离 ==========
print(
    f"\n=== Training Decoder with {model_info['type'].upper()} Embeddings ===")

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(
    range(data.edge_index.shape[1]),
    test_size=0.2,
    random_state=42
)

train_edge_attr = data.edge_attr[train_idx]
test_edge_attr = data.edge_attr[test_idx]

print(f"Training edges: {len(train_idx)}")
print(f"Test edges: {len(test_idx)}")

# 初始化解码器
try:
    from model import GCNDecoder
    print("GCNDecoder imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please check if model.py exists and contains GCNDecoder class")

# ========== 关键修改：使用ESM嵌入维度 ==========
input_dim = embedding_dim  # 1280 (ESM嵌入维度)，不再是num_hidden
out_feats = 512  # 中间层维度

print(f"Decoder input dimension: {input_dim}")
print(f"Decoder output features: {out_feats}")

decoder = GCNDecoder(input_dim=input_dim, out_feats=out_feats).to(device)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.0005, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=100, gamma=0.9)

# 训练解码器
print("\n=== Training Decoder with  Embeddings ===")
best_train_loss = float('inf')
start = t()

for epoch in range(1, 6001):  # 1500轮训练
    decoder.train()
    decoder_optimizer.zero_grad()

    # 预测训练集边属性
    predicted_edge_attr = decoder(data.x, data.edge_index[:, train_idx])
    train_loss = F.mse_loss(predicted_edge_attr, train_edge_attr)

    train_loss.backward()
    decoder_optimizer.step()
    scheduler.step()

    # 每50轮评估一次
    if epoch % 50 == 0:
        decoder.eval()
        with torch.no_grad():
            # 测试集评估
            test_predicted = decoder(data.x, data.edge_index[:, test_idx])
            test_loss = F.mse_loss(test_predicted, test_edge_attr)
            test_mae = mean_absolute_error(test_edge_attr.cpu().numpy(),
                                         test_predicted.cpu().numpy())
            test_r2 = r2_score(test_edge_attr.cpu().numpy(),
                             test_predicted.cpu().numpy())

        now = t()
        print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | '
              f'Test Loss: {test_loss:.4f} | MAE: {test_mae:.4f} | '
              f'R²: {test_r2:.4f} | Time: {now - start:.2f}s')

        # 保存最佳模型
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(decoder.state_dict(), 'best_esm_decoder.pth')

# 最终评估
print("\n=== Final Evaluation with ESM Embeddings ===")
decoder.eval()
with torch.no_grad():
    test_predicted = decoder(data.x, data.edge_index[:, test_idx])
    test_loss = F.mse_loss(test_predicted, test_edge_attr)
    test_mae = mean_absolute_error(test_edge_attr.cpu().numpy(),
                                 test_predicted.cpu().numpy())
    test_r2 = r2_score(test_edge_attr.cpu().numpy(),
                     test_predicted.cpu().numpy())

print("\n=== Final Results with ESM Embeddings ===")
print(f'Test MSE: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')
print(f'Test R²: {test_r2:.4f}')

# 额外分析：比较真实值和预测值
print(f"\n=== Prediction Analysis ===")
test_real = test_edge_attr.cpu().numpy()
test_pred = test_predicted.cpu().numpy()

print(f"Real values range: [{test_real.min():.4f}, {test_real.max():.4f}]")
print(f"Predicted values range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
print(f"Mean absolute difference: {np.mean(np.abs(test_real - test_pred)):.4f}")

# 保存最终结果
results_df = pd.DataFrame({
    'real_distance': test_real.flatten(),
    'predicted_distance': test_pred.flatten(),
    'absolute_error': np.abs(test_real - test_pred).flatten()
})

# results_df.to_csv('esm_decoder_predictions.csv', index=False)
# print("Prediction results saved to 'esm_decoder_predictions.csv'")

# print(f"\n🎉 ESM嵌入向量抗原距离预测完成！")
# print(f"📊 使用了1280维ESM嵌入替代原始特征")
print(f"🔬 解码器成功预测抗原距离")