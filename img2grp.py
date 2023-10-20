import cv2
import torch
import os

from torchvision import transforms
from torch_geometric.data import Data
from swin import swin_base_patch4_window7_224_in22k
from torch_geometric.nn import knn_graph

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############################################### PATCH ##########################################################
def img2graph1(img_path, img_filename, save_folder, swin_model):
    try:
        image = cv2.imread(img_path)
            
        if image is None:
            raise FileNotFoundError(f"Image file '{img_path}' not found or cannot be read.")
            
        if image.shape[:2] != image_size:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, image_size)

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

                # 对图像进行数据转换
        input_image = data_transform(image).unsqueeze(0)

        with torch.no_grad():
            features = swin_model.patch_embed(input_image)
            # print(features[0])

            # 获取特征张量的形状
            nodes_features = features[0]
            num_nodes, num_features = nodes_features.shape[1], nodes_features.shape[2]
                
            # 定义每个节点包含的连续 patch 的数量
            num_patches_node = 8 * 8
            # # 重塑特征张量以创建节点
            num_nodes = nodes_features.shape[1] // num_patches_node
            nodes_features = nodes_features.view(1, num_nodes, num_patches_node, -1)
            # # nodes_features = nodes_features.mean(2)
            nodes_features = nodes_features.permute(0, 2, 1, 3).contiguous().view(1, -1, nodes_features.shape[-1])  #节点特征concat(也可以取平均值)
            

            ############################    Data   ###############################
            graph_label = torch.tensor([3], dtype=torch.long)  # 3表示类别标签
            edge_index = knn_graph(nodes_features.view(num_nodes, -1), k=k, batch=None, loop=False)

            # create the graph of torch_geometric
            data = Data(x=nodes_features.view(num_nodes, -1), edge_index=edge_index, y=graph_label)    # node = 9, feature dim of node = 90000, start+end = 2, num of edge = 17, label num of node = 9
            # print(data.edge_index.t())
            # 生成文件名并保存Data对象
            filename = os.path.splitext(img_filename)[0]  # 使用图像文件名作为文件名
            save_path = os.path.join(save_folder, f'{filename}.pt')
            torch.save(data, save_path)
            print(f'Data object saved to {save_path}')

    except Exception as e:
        print(f"An error occurred while processing '{img_filename}': {e}")


############################################### WINDOWS ##########################################################
def img2graph2(img_path, img_filename, save_folder, swin_model):
    try:
        image = cv2.imread(img_path)
            
        if image is None:
            raise FileNotFoundError(f"Image file '{img_path}' not found or cannot be read.")
            
        if image.shape[:2] != image_size:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, image_size)

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

                # 对图像进行数据转换
        input_image = data_transform(image).unsqueeze(0)

        with torch.no_grad():
            features = swin_model.patch_embed(input_image)
            # print(features[0])

            # 获取特征张量的形状
            nodes_features = features[0]
            num_nodes, num_features = nodes_features.shape[1], nodes_features.shape[2]
                
            # 定义每个节点包含的连续 patch 的数量
            num_patches_node = 8 * 8
            # # 重塑特征张量以创建节点
            num_nodes = nodes_features.shape[1] // num_patches_node
            nodes_features = nodes_features.view(1, num_nodes, num_patches_node, -1)
            # # nodes_features = nodes_features.mean(2)
            nodes_features = nodes_features.permute(0, 2, 1, 3).contiguous().view(1, -1, nodes_features.shape[-1])  #节点特征concat(也可以取平均值)
            

            ############################    Data   ###############################
            graph_label = torch.tensor([3], dtype=torch.long)  # 3表示类别标签
            edge_index = knn_graph(nodes_features.view(num_nodes, -1), k=k, batch=None, loop=False)

            # create the graph of torch_geometric
            data = Data(x=nodes_features.view(num_nodes, -1), edge_index=edge_index, y=graph_label)    # node = 9, feature dim of node = 90000, start+end = 2, num of edge = 17, label num of node = 9
            # print(data.edge_index.t())
            # 生成文件名并保存Data对象
            filename = os.path.splitext(img_filename)[0]  # 使用图像文件名作为文件名
            save_path = os.path.join(save_folder, f'{filename}.pt')
            torch.save(data, save_path)
            print(f'Data object saved to {save_path}')

    except Exception as e:
        print(f"An error occurred while processing '{img_filename}': {e}")


#####################################################################################################################
def main():
    folder_path = '/Users/tianjiexin/Downloads/thesis/code/swin_transformer/dataset/MRI/test/pituitary'
    save_folder = '/Users/tianjiexin/Downloads/thesis/code/gnn/MyGraph/MRI/test/pituitary'  # 替换为您希望保存的文件夹路径
    os.makedirs(save_folder, exist_ok=True)

    # Swin Transformer模型
    swin_model = swin_base_patch4_window7_224_in22k(num_classes=num_classes, image_size=image_size)

    # 加载Swin Transformer的预训练权重
    swin_checkpoint = torch.load('/Users/tianjiexin/Downloads/thesis/code/swin_transformer/weights/modelv1-2_mri.pth', map_location='cpu')
    swin_model.load_state_dict(swin_checkpoint)
    swin_model.to(device)
    swin_model.eval()

    for img_filename in os.listdir(folder_path):
        if img_filename.endswith((".jpg", ".JPG", ".png", ".PNG", "jpeg")):
            img_path = os.path.join(folder_path, img_filename)
            img2graph1(img_path, img_filename, save_folder, swin_model)

if __name__ == "__main__":
    image_size = (224, 224)
    num_classes = 4
    k = 8

    main()

# graph_path = '/Users/tian/Downloads/thesis/code/gnn/MyGraph/CT/test/squamouscellcarcinomalefthilum/000111.pt'
# data = torch.load(graph_path)
# print(data)     # Data(x=[9, 86700], edge_index=[2, 24], y=[1])


# MyGraph_train =         MyGraph_test = 1311
