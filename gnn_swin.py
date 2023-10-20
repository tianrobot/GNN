import torch
import os
import cv2

import torch.nn.functional as F
import numpy as np

from torchvision import transforms
from PIL import Image
from torch_geometric.nn import global_mean_pool, global_max_pool
from swin import swin_base_patch4_window7_224_in22k
from torch_geometric.data import Data, DataLoader
from sklearn.cluster import KMeans
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv, TopKPooling, SAGEConv, knn_graph
from sklearn.neighbors import NearestNeighbors

# 设定参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = 224
num_classes = 4
k = 7
input_folder = '/Users/tianjiexin/Downloads/thesis/code/swin_transformer/dataset/MRI/train/glioma'
save_folder = '/Users/tianjiexin/Downloads/thesis/code/gnn/MyGraph/MRI/train/glioma'
os.makedirs(save_folder, exist_ok=True)


# Swin Transformer模型
swin_model = swin_base_patch4_window7_224_in22k(num_classes=num_classes, image_size=image_size)
# 加载Swin Transformer的预训练权重
swin_checkpoint = torch.load('/Users/tianjiexin/Downloads/thesis/code/swin_transformer/weights/modelv1-2_mri.pth', map_location='cpu')
swin_model.load_state_dict(swin_checkpoint)
swin_model.to(device)
swin_model.eval()
# print(swin_model)


# 定义图像
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])


# 遍历输入文件夹中的图像文件并处理
for filenam in os.listdir(input_folder):
     if filenam.endswith(('.jpg', '.png', 'jpeg')):
          image_path = os.path.join(input_folder, filenam)
          # 加载并处理图像
          with torch.no_grad():
               image = Image.open(image_path).convert("RGB")
               image = transform(image).unsqueeze(0).to(device)
               x, H, W = swin_model.patch_embed(image)
               x = swin_model.pos_drop(x)
               for layer in swin_model.layers:
                    x, H, W = layer(x, H, W)
               x = swin_model.norm(x)

               # 获取节点特征
               node_features = x.squeeze().cpu().numpy()  # Convert tensor to NumPy array
               num_nodes = node_features.shape[0]

               # 使用K近邻算法构建边，这里选择K=16
               knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
               knn.fit(node_features)
               distances, neighbors = knn.kneighbors(node_features)

               # 构建边索引 edge_index
               edge_index = []
               for i in range(num_nodes):
                   for j in neighbors[i]:
                       if i != j:
                            # 添加边索引，注意PyG中边索引是一个大小为[2, num_edges]的LongTensor
                            edge_index.append([i, j])
               edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()


               # 构建标签 y
               y = torch.tensor([0], dtype=torch.float)  # 替换成您的标签数据

               # 创建Data对象
               data = Data(x=torch.tensor(node_features, dtype=torch.float),
                           edge_index=edge_index, y=y)

               # 保存Data对象
               filename = os.path.splitext(filenam)[0]  # 使用图像文件名作为文件名
               save_path = os.path.join(save_folder, f'{filename}.pt')
               torch.save(data, save_path)
               print(f'Data object saved to {save_path}')

# data = torch.load('graph_data.pt')
# print(data)








# data_list = [] 

# image_path = '/Users/tianjiexin/Downloads/thesis/code/swin_transformer/dataset/MRI/train/glioma/Tr-gl_0012.jpg'
# save_path = '/Users/tianjiexin/Desktop'
# grp_path = '/Users/tianjiexin/Downloads/thesis/code/gnn/MyGraph/MRI/train/glioma'
# image = Image.open(image_path).convert('RGB')


# data_transform = transforms.Compose([
#     transforms.Resize((image_size, image_size)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # 对图像进行数据转换
# input_image = data_transform(image).unsqueeze(0) # type: ignore


# with torch.no_grad():
#     features = swin_model.patch_embed(input_image)
#     # print(features[0])
#     # features = swin_model(input_image)
#     # print(features[0])

#     # print(features[0].shape)    # torch.Size([1, 3136, 128]) 有 3136 个向量，每个向量有 128 维
#     # 获取特征张量的形状
#     nodes_features = features[0]
#     num_nodes, num_features = nodes_features.shape[1], nodes_features.shape[2]
#     # print(num_nodes)
#     # print(num_features)

# # 定义每个节点包含的连续 patch 的数量
# # num_patches_node = 7
# # # 重塑特征张量以创建节点
# # num_nodes = nodes_features.shape[1] // num_patches_node
# # nodes_features = nodes_features.view(1, num_nodes, num_patches_node, -1)
# # # nodes_features = nodes_features.mean(2)
# # nodes_features = nodes_features.permute(0, 2, 1, 3).contiguous().view(1, -1, nodes_features.shape[-1])  #节点特征concat(也可以取平均值)


# graph_label = torch.tensor([0], dtype=torch.long)
# edge_index = knn_graph(nodes_features.view(num_nodes, -1), k=k, batch=None, loop=False)

# data = Data(x=nodes_features.view(num_nodes, -1), edge_index=edge_index, y=graph_label) 
# torch.save(data, os.path.join(save_path, 'data.pt'))
# loaded_data = torch.load(os.path.join(grp_path, 'Tr-gl_0012.pt'))
# print(loaded_data)
# # print(loaded_data.x)
# # print(loaded_data.edge_index)
# # print(loaded_data.y)

