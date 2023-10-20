import torch
import os
import cv2 
from torch.utils.data import Dataset
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch_geometric.data import Batch
import torch.optim as optim
import torch.nn.functional as F

image_path = 'Tr-gl_0012.jpg'
img = cv2.imread(image_path)
# print(img.shape)   # [h, w, c] = (512, 512, 3)
img_tensor = torch.tensor(img, dtype=torch.float32)

height, width, channels = img.shape
feature_dim = height * width * channels
print(f"图像的特征维度: {feature_dim}")     # 786432
# print(img_tensor.shape) # ([512, 512, 3])





