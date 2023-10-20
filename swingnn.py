import os
import torch
from swin import swin_base_patch4_window7_224_in22k
from torchvision import transforms
from PIL import Image
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv, TopKPooling, SAGEConv, knn_graph

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_size = 224
data_root = '/Users/tianjiexin/Downloads/thesis/code/swin_transformer/dataset/MRI/train/glioma/Tr-gl_1292.jpg'

# Swin Transformer模型
swin_model = swin_base_patch4_window7_224_in22k(num_classes=4, image_size=image_size)

# 加载Swin Transformer的预训练权重
swin_checkpoint = torch.load('/Users/tianjiexin/Downloads/thesis/code/swin_transformer/weights/modelv1-2_mri.pth', map_location='cpu')
swin_model.load_state_dict(swin_checkpoint)

swin_model.to(device)
swin_model.eval()

k_neighbors = 8  # 设置k值，即每个节点的最近邻数

data_list = []


# 遍历四个子文件夹
# subfolders = [folder for folder in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, folder))]
# for i, folder in enumerate(subfolders):
#     folder_path = os.path.join(data_root, folder)
    
#     image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # for image_path in image_files:
image = Image.open(data_root).convert('RGB')

data_transform = transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])

input_image = data_transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = swin_model(input_image)
    # print(output.shape)
    print(swin_model)
            # features = swin_model.patch_embed(input_image)
            # nodes_features = features[0]
            # # num_nodes, num_features = nodes_features.shape[1], nodes_features.shape[2]

            # # 定义每个节点包含的连续 patch 的数量
            # num_patches_node = 7 * 7
            # # 重塑特征张量以创建节点
            # num_nodes = nodes_features.shape[1] // num_patches_node
            # nodes_features = nodes_features.view(1, num_nodes, num_patches_node, -1)
            # nodes_features = nodes_features.permute(0, 2, 1, 3).contiguous().view(1, -1, nodes_features.shape[-1])
            # nodes_features = nodes_features.mean(2)

        # 计算图的边
        # edge_index = compute_edges(nodes_features)
        # edge_index = knn_graph(nodes_features.view(num_nodes, -1), k=k_neighbors, batch=None, loop=False)

        # # 创建Data对象，并设置特征、边和标签
        # graph_label = torch.tensor([i], dtype=torch.long)

        # data = Data(x=nodes_features.view(num_nodes, -1), edge_index=edge_index, y=graph_label)

        # data_list.append(data)

# 保存所有Data对象到一个.pt文件
# save_path = '/Users/tianjiexin/Desktop'
# torch.save(data_list, os.path.join(save_path, 'train.pt'))


# 读取保存的Data对象
# loaded_data_list = torch.load(os.path.join(save_path, 'train.pt'))
# for data in loaded_data_list:
#     print(data)     # Data(x=[64, 6272], edge_index=[2, 512], y=[1])
