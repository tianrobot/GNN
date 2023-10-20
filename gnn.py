import cv2
import numpy as np
import torch_geometric
import torch.nn.functional as F
import torch.optim as optim
import os 
import sys
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import Linear, Dropout
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv, TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Batch
from typing import Callable, cast
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


#######################################     Dataset     ######################################
class GraphDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = []

        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.endswith(".pt"):
                        file_path = os.path.join(subdir_path, filename)
                        self.file_paths.append(file_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        data = torch.load(file_path)
        return data
    
    @staticmethod
    def custom_collate(batch):
        return Batch.from_data_list(batch)

# train_path = '/Users/tianjiexin/Desktop/train.pt'
# train_data = torch.load(train_path)

# test_path = '/Users/tianjiexin/Desktop/test.pt'
# test_data = torch.load(test_path)

#######################################     Model     ######################################
class GNNmodel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_heads=4):
        super(GNNmodel, self).__init__()
        self.conv1 = GATv2Conv(num_features, 128, heads=num_heads)
        self.conv2 = GATv2Conv(128 * num_heads, 64, heads=num_heads)
        self.conv3 = GATv2Conv(1280, 32, heads=num_heads)

        # self.conv1 = GATConv(num_features, 128, heads=num_heads)
        # self.conv2 = GATConv(128 * num_heads, 64, heads=num_heads)
        # self.conv3 = GATConv(64 * num_heads,32, heads=num_heads)
      
        # self.conv1 = GCNConv(num_features, 128)
        # self.conv2 = GCNConv(128, 64)
        # self.conv3 = GCNConv(64, 32)

        self.bn1 = torch.nn.BatchNorm1d(128 * num_heads)
        self.bn2 = torch.nn.BatchNorm1d(64 * num_heads)
        self.bn3 = torch.nn.BatchNorm1d(32 * num_heads)
        
        # self.fc = nn.Linear(32 * num_heads, num_classes)
        # self.fc = nn.Linear(32, num_classes)

        # Define MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(1920, hidden_dim),  # Input size is 9088 from the previous layers
            nn.ReLU(),
            nn.Linear(256, 128),  # Input size is 9088 from the previous layers
            nn.ReLU(),
            # nn.Linear(32, 32),  # Input size is 9088 from the previous layers
            # nn.ReLU(),
            # nn.Linear(64, 32),  # Input size is 9088 from the previous layers
            # nn.ReLU(),
            nn.Linear(128, num_classes)  # Output size is num_classes
        )

    def forward(self, data):
        x0, edge_index = data.x, data.edge_index
        # print(x0.shape) # torch.Size([1568, 8192])

        # Layer 1
        x1 = F.relu(self.conv1(x0, edge_index))
        # print(x1.shape) # torch.Size([1568, 512])
        # x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.bn1(x1)
        # print(x1.shape) # torch.Size([1568, 512])

        # Layer 2
        x2 = F.relu(self.conv2(x1, edge_index))
        # print(x2.shape) # torch.Size([1568, 256])
        # x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.bn2(x2)
        # print(x2.shape) # torch.Size([1568, 256])
        x0_x2 = torch.cat((x0, x2), dim=-1)  # Residual connection
        # print(x0_x2.shape)  # torch.Size([1568, 8448])

        # Layer 3
        x3 = F.relu(self.conv3(x0_x2, edge_index))
        # print(x3.shape) # torch.Size([1568, 128])
        # x3 = F.dropout(x3, p=0.5, training=self.training)
        x3 = self.bn3(x3)  
        # print(x3.shape) # torch.Size([1568, 128])
        x0_x1_x2_x3 = torch.cat((x0, x1, x2, x3), dim=-1)
        # print(x0_x1_x2_x3.shape)    # torch.Size([1568, 9088])

        # Residual Connection from Layer 3 to Pooling
        batch = torch.zeros(data.x.shape[0], dtype=int) if data.batch is None else data.batch
        x_pool = global_mean_pool(x0_x1_x2_x3, batch)
        x_pool = F.dropout(x_pool, p=0.5, training=self.training)
        # Final classification layer
        x = self.mlp(x_pool).squeeze(1)

        return x
    
#######################################     Training     ######################################
def train():
    optimizer = optim.Adam(model.parameters(), lr=5e-05, betas=(0.9,0.9999), eps=1e-08, weight_decay=1e-4)
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    num_epochs = 20
    
    for epoch in range(num_epochs):
        accu_loss = 0.0  # Cumulative loss
        correct_predictions = 0  # Cumulative number of correctly predicted samples
        total_samples = 0

        for data in tqdm(train_data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            graph_labels = data.y.long()
            # print(output.shape, graph_labels.shape)
            # predicted_labels = torch.max(output, dim=1)[1]
            predicted_labels = torch.argmax(output, dim=1)
            correct_predictions += (predicted_labels == graph_labels).sum().item()
            total_samples += len(data)

            loss = loss_function(output, graph_labels)
            loss.backward()
            optimizer.step()
            
            accu_loss += loss.item()

        avg_loss = accu_loss / len(train_data_loader)
        accuracy = correct_predictions / total_samples

        print(f'Epoch [{epoch+1}] | Loss: {avg_loss:.3f} | Acc: {accuracy:.3f}')

        # Save model weights
        torch.save(model.state_dict(), f'Weights/SwinGNN-{epoch+1}_mri.pth')


#######################################     Testing     ######################################
# 定义性能评估函数
def evaluate_model(model, test_data_loader):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in tqdm(test_data_loader):
            data = data.to(device)
            output = model(data)
            graph_labels = data.y.long()

            # predicted_labels = torch.max(output, dim=1)[1]
            predicted_labels = torch.argmax(output, dim=1)
            graph_labels = graph_labels.squeeze()

            correct_predictions += (predicted_labels == graph_labels).sum().item()
            total_samples += len(data)

            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(output, graph_labels)
            test_loss += loss.item()

    accuracy = correct_predictions / total_samples
    avg_test_loss = test_loss / len(test_data_loader)
    return accuracy, avg_test_loss

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = GraphDataset(root_dir='/Users/tianjiexin/Downloads/thesis/code/gnn/MyGraph/MRI/train')
    train_data_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True, collate_fn=train_dataset.custom_collate)

    test_dataset = GraphDataset(root_dir='/Users/tianjiexin/Downloads/thesis/code/gnn/MyGraph/MRI/test')
    test_data_loader = DataLoader(test_dataset, batch_size= 32, shuffle=False, collate_fn=test_dataset.custom_collate)

    model = GNNmodel(num_features=1024, hidden_dim=256, num_classes=4).to(device)
    # print(model)
    
    # train()

    # test
    model.load_state_dict(torch.load('Weights/SwinGNN-6_mri.pth'))
    test_accuracy, test_loss = evaluate_model(model, test_data_loader)
    print(f'Loss: {test_loss:.3f} Acc: {test_accuracy:.4f}')
