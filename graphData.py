from sklearn import preprocessing
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import random


class graphDataset(Dataset):#原本353条边无向图变成706条边
    def __init__(self,subtype):
        path = f"data/{subtype}_antigenicDistance.xlsx"
        strain = pd.read_excel(path)#数据集信息赋值给strain
        # **存储原始的毒株名称**
        # 保留原始名称并创建小写版本
        strain["strain1_original"] = strain["strain1"].copy()
        strain["strain2_original"] = strain["strain2"].copy()
        strain["strain1"] = strain["strain1"].str.lower()
        strain["strain2"] = strain["strain2"].str.lower()
        # strain["strain1"] = strain["strain1"].astype(str).str.lower()#更换师兄数据集后报错，用这两行
        # strain["strain2"] = strain["strain2"].astype(str).str.lower()
        # 建立小写到原始名称的映射字典
        unique_names = pd.concat([
            strain[["strain1", "strain1_original"]].rename(
                columns={"strain1": "lower", "strain1_original": "original"}),
            strain[["strain2", "strain2_original"]].rename(
                columns={"strain2": "lower", "strain2_original": "original"})
        ]).drop_duplicates()
        self.lower_to_original = pd.Series(unique_names['original'].values,
                                           index=unique_names[
                                               'lower']).to_dict()
        lables = preprocessing.LabelEncoder()
        unique_labels = np.unique(
            list(strain["strain1"]) + list(strain["strain2"]))
        lables.fit(unique_labels)
        #LabelEncoder会发现4个独特的标签：A1, A2, B1, B2。
        #它会给每个病毒株名称分配一个唯一的整数索引。例如，可能是A1 -> 0, A2 -> 1, B1 -> 2, B2 -> 3。
        strain["id1"] = lables.transform(list(strain["strain1"].values))
        strain["id2"] = lables.transform(list(strain["strain2"].values))
        #将strain1和strain2列中的所有病毒株名称转换为对应的整数索引
        strain.sort_values(by=['id1', 'id2'],inplace=True)
        #根据这些新生成的数值索引（id1和id2）对DataFrame进行排序sort
        "DATA edge"
        oriN = strain["id1"].values
        endN = strain["id2"].values#起始节点终止节点信息
        on = np.hstack((oriN,endN))#np.hstack 来自 NumPy 库，它的作用是将多个数组水平（按列顺序）堆叠起来
        en = np.hstack((endN,oriN))#边信息，无向图
        # edge_index = torch.tensor(np.array([on,en]), dtype=torch.long)
        edge_index = torch.tensor(np.array([oriN, endN]), dtype=torch.long)#张量，包含了图中所有边的连接信息
        "edge_attr"                    #oriN在第一行，endN在第二行
        # ea = np.hstack((strain["distance"].values,strain["distance"].values))
        # ea = strain["distance"].values
        #这里面存储了抗原距离假设有3组毒株，抗原距离那一列是10，20，30则ea=[10, 20, 30, 10, 20, 30]#前3个是原始索引每个节点的距离，后3个是无向图复刻边的索引
        # ea_min = np.min(ea)
        # shifted_arr = ea - ea_min + 1#所有抗原距离值，减去最小值，加1确保减去最小值后，最小值不为0，至少为1
        # ea_min = np.min(shifted_arr)
        # #再次计算调整后数组的最小值
        # ea_max = np.max(shifted_arr)
        # #计算调整后数组的最大值
        # ea_normalized = (shifted_arr - ea_min) / (ea_max - ea_min)#保证范围严格为[0, 1]，- ea_min是为了可以确保最小值完全等于0
        # edge_attr  = torch.tensor(ea_normalized, dtype=torch.float)
        # 将归一化后的数组转换为PyTorch张量
        edge_attr = torch.tensor(strain["distance"].values, dtype=torch.float)
        # 未进行归一化
        "x"
        provect = pd.read_csv("data/protVec_100d_3grams.csv", delimiter='\t')
        # strain = pd.read_csv(path3, names=['seq', 'description'])
        # strain["description"] = strain["description"].str.lower()
        trigrams = list(provect['words'])
        trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
        trigram_vecs = provect.loc[:, provect.columns != 'words'].values
        #从provect DataFrame中提取除words列之外的所有列。这些列包含了每个三元组对应的向量表示。
        # .values将DataFrame的这部分转换为一个NumPy数组，方便后续操作和向量检索
        x = []
        virus_names = []  # 初始化病毒名称列表
        #初始化一个空列表x，用来存储后续处理的向量化蛋白质序列
        for strain_name in lables.classes_:#病毒株名称存储在lables.classes_里面
            if len(strain[strain["strain1"]==strain_name]["seq1"])>0:
                seq = strain[strain["strain1"]==strain_name]["seq1"].values[0]
            # 这行代码首先检查DataFrame strain 中 strain1 列是否包含当前迭代的病毒株名称（strain_name）。
            # 如果strain_name存在于strain1列中，则继续检查对应的seq1列
            # （存储相关的蛋白质序列）是否包含数据。len()函数用来确认在seq1列中是否存在为这
            # 个病毒株名称记录的序列，即这个条目是否为空或存在
            else:#
                seq = strain[strain["strain2"] == strain_name]["seq2"].values[0]
                #如果当前病毒株名称不在strain1列中，那么它一定在strain2列中
            strain_embedding = []#初始化一个空列表来存储当前病毒株的序列的向量表示
            for i in range(0, len(seq) - 2):#遍历序列，从头到尾部前两个字符的位置，以便能够抽取长度为3的三元组
                #len(seq) - 2是564，i最大到563
                trigram = seq[i:i + 3]#从序列中抽取长度为3的片段，这是一个三元组
                if "-" in trigram:#检查三元组中是否包含"-"字符。这通常表示序列的一个位置上的氨基酸是未知的
                    tri_embedding = trigram_vecs[trigram_to_idx['<unk>']]
                    
                else:
                    # tri_embedding = trigram_vecs[trigram_to_idx[trigram]]
                    # #如果三元组中的氨基酸都已知，从向量表trigram_vecs中检索出对应的向量
                    #更换数据集后有1个3元组字典中不存在
                    try:
                        tri_embedding = trigram_vecs[trigram_to_idx[trigram]]
                strain_embedding.append(tri_embedding)
                #将检索到的向量加入到病毒株的向量列表中
            x.append(strain_embedding)#将处理完的病毒株向量列表添加到主列表x中，该列表最终将包含所有病毒株的向量表示
            # 列表中中的每一项代表一个566*100维的向量
            original_name = self.lower_to_original.get(strain_name, strain_name)
            virus_names.append(original_name)  #正确获取原始名称
        x = np.array(x)#将列表转换为NumPy数组，为了进一步处理
        x = torch.tensor(x.reshape(x.shape[0],-1), dtype=torch.float)#只有x.shape[0]代表行数和-1代表自动计算，这两个，所以将n维数组转化为2维数组
        self.data = Data(x=x,edge_index=edge_index,edge_attr=edge_attr, virus_names=virus_names)
# gData = graphDataset("jiaH1N1")  # 's' 应该是一个表示亚型的字符串
# print("Loaded graph data with shapes:")
# print("Node features (x) shape:", gData.data.x.shape)
# print("Edge index shape:", gData.data.edge_index.shape)
# print("Edge attributes (edge_attr) shape:", gData.data.edge_attr.shape)
#H1N1列表中strain1和strain2列中共有50个不同的病毒株名称
#Node features (x) shape: torch.Size([50, 56400])
# Edge index shape: torch.Size([2, 706])
# Edge attributes (edge_attr) shape: torch.Size([706])这个数据已经是归一化后的了

