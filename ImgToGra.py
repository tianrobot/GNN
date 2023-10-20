import cv2
import torch
import os

from torchvision import transforms
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def img2graph1(img_path, img_filename, save_folder):
    try:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            raise FileNotFoundError(f"Image file '{img_path}' not found or cannot be read.")

        desired_size = (224, 224)
        if image.shape[:2] != desired_size:
            image = cv2.resize(image, desired_size)  # 调整图像尺寸

        patch_size = 32  # 你的 patch 大小
        num_patches = 7  # 7x7 的划分

        patches = []
        for i in range(num_patches):
            for j in range(num_patches):
                patch = image[i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches.append(patch)

        # 对图像进行数据转换
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        patch_tensors = [data_transform(patch).unsqueeze(0) for patch in patches]

        # 将 patch tensors 拼接成节点特征
        nodes_features = torch.cat(patch_tensors, dim=0)
        num_nodes, num_features, _, _ = nodes_features.shape

        graph_label = torch.tensor([0], dtype=torch.long)  # 类别标签
        edge_index = knn_graph(nodes_features.view(num_nodes, -1), k=16, batch=None, loop=False)

        # 创建 torch_geometric 的 Data 对象
        data = Data(x=nodes_features.view(num_nodes, -1), edge_index=edge_index, y=graph_label)

        # 生成文件名并保存 Data 对象
        filename = os.path.splitext(img_filename)[0]  # 使用图像文件名作为文件名
        save_path = os.path.join(save_folder, f'{filename}.pt')
        torch.save(data, save_path)
        print(f'Data object saved to {save_path}')

    except Exception as e:
        print(f"An error occurred while processing '{img_filename}': {e}")

def main():
    folder_path = '/Users/tianjiexin/Downloads/thesis/code/swin_transformer/dataset/CT/test/non-COVID'
    save_folder = '/Users/tianjiexin/Downloads/thesis/code/gnn/MyGraph/CT/test/non-COVID'
    os.makedirs(save_folder, exist_ok=True)

    for img_filename in os.listdir(folder_path):
        if img_filename.endswith((".jpg", ".JPG", ".png", ".PNG", "jpeg")):
            img_path = os.path.join(folder_path, img_filename)
            img2graph1(img_path, img_filename, save_folder)

if __name__ == "__main__":
    main()
    # graph_path = '/Users/tianjiexin/Downloads/thesis/code/gnn/MyGraph/CT/test/non-COVID'
    # loaded_data = torch.load(os.path.join(graph_path, 'Non-Covid(5).pt'))
    # print(loaded_data.x[47])
