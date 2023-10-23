import os
import torch 

import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(data, save_folder):
    G = nx.Graph()

    # Add nodes to the NetworkX graph
    for node_id in range(data.num_nodes):
        G.add_node(node_id)

    # Add edges to the NetworkX graph
    edge_list = data.edge_index.t().tolist()
    for edge in edge_list:
        src, dst = edge
        G.add_edge(src, dst)

    # Create a layout for the nodes (e.g., using Kamada-Kawai layout)
    layout = nx.kamada_kawai_layout(G)

    # Draw the NetworkX graph using Matplotlib
    plt.figure(figsize=(8, 8))
    nx.draw(G, layout, with_labels=True, node_size=100, font_size=8)
    plt.title("Visualization of the Graph")
    plt.axis('off')  # Turn off axis labels
    plt.savefig(os.path.join(save_folder, 'graph_visualization.png'))  # Save the visualization as an image
    plt.show()


graph_path = '/Users/tianjiexin/Downloads'
data = torch.load(os.path.join(graph_path, 'XR.pt'))
save_folder = '/Users/tianjiexin/Downloads'

visualize_graph(data, save_folder)