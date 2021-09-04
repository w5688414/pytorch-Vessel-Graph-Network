import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from joblib.numpy_pickle_utils import xrange
import pickle

from model import vgn_inference_module
import VGG16
from vertex_sampling_and_edge import vgn_feature
from dataset import Datasets
from GAT_tf1 import GraphAttentionLayer
from tqdm import tqdm 
'''
save the vertex sampling and edge construction to a file and using the pickle library
'''

# Base directory
# base_dir = "C:\\Users" \
#        "\\VGN_v5\\"
# training
split='test'
base_dir='/media/data/projects/VGN/VGN_workingcopy'
# We save the graphs for all training images to a file in the training directory
graph_dir = os.path.join(base_dir,"DRIVE/{}".format(split))
# We use the cpu
device = torch.device("cuda")
model = VGG16.Net(num_classes = 1)
criterion = nn.BCELoss()
data_path = os.path.join(base_dir,"DRIVE/{}/".format(split))
delta = 16
dist_thresh = 40
graph_file = os.path.join(graph_dir, "graphs_{}_delta".format(split) + str(delta) +
                                     "_dist_thresh" + str(dist_thresh) + ".pickle")

train_data = Datasets(path=data_path)
# We set batch_size to 1 to make things simpler when applying the model/creating a graph for a single image
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)

# Load best model - the name of your model might be "best_model.pth"
best_model_path = os.path.join(base_dir,"best_model_Aug10.pth")
model.load_state_dict(torch.load(best_model_path))
print("Model loaded.")

# We create vertices and edges for im based on prob_map (converted to a Numpy array)
# We store the graphs in a dict, whose keys will be the name of each image
graphs_dict = {}

if not os.path.exists(graph_file):
    print("Creating graphs for training images..")
    # Load images and labels and create a graph for each image
    for i, train_input in tqdm(enumerate(train_loader)):
        # Apply the pre-trained CNN model to the example image to get the probability map, intermediate CNN features,
        # and the concatenated feature matrix of the CNN module
        prob_map, cnn_feat, concat_feat = model(train_input['img'])

        # prob_map will be used to create the vertices and edges for the GNN module
        #print("prob_map shape: " + str(prob_map.shape))
        # cnn_feat will be used in the inference module
        #print("cnn_feat[1] shape: " + str(cnn_feat[1].shape))
        #print("cnn_feat[2] shape: " + str(cnn_feat[2].shape))
        #print("cnn_feat[4] shape: " + str(cnn_feat[4].shape))
        #print("cnn_feat[8] shape: " + str(cnn_feat[8].shape))
        # concat_feat will be used to create the input features for each vertex for the GNN module
        #print("concat_feat shape: " + str(concat_feat.shape))

        print("Image index: " + str(i))
        print("Creating graph with delta=" + str(delta) + " and dist_thresh=" + str(dist_thresh))
        tmp_graph = vgn_feature(prob_map.detach().numpy(), delta, dist_thresh, base_dir)
        # Print out number of nodes and edges
        print("Graph created.")
        print("no. of nodes: " + str(tmp_graph.number_of_nodes()))
        print("no. of edges: " + str(tmp_graph.number_of_edges()))
        graphs_dict[train_input['img_name'][0]] = tmp_graph

    # Save the graphs list to file
    with open(graph_file, 'wb') as f:
        pickle.dump(graphs_dict, f)
    print("Graphs created for training dataset and saved to file: " + graph_file)
    # We use an example graph to visualize the results
    graph = tmp_graph
else:
    with open(graph_file, 'rb') as f:
        graphs_dict = pickle.load(f)

    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
    train_input = next(iter(train_loader))
    graph = graphs_dict[train_input['img_name'][0]]
    prob_map, cnn_feat, concat_feat = model(train_input['img'][0].unsqueeze(0))

# Code to plot prob_map, drawing arrows where connected nodes are in the image
fig = plt.figure()
im = plt.imshow(np.squeeze(prob_map.detach().numpy()))
fig.colorbar(im)
nodelist = graph.nodes
num_nodes = 0
print("plotting edges on probability map..")
for (i,n) in graph.adjacency():
    if len(n) > 0:
        tmp_node1 = nodelist[i]
        x1 = tmp_node1['x']
        y1 = tmp_node1['y']
        for i_node in n.keys():
            tmp_node2 = nodelist[i_node]
            x2 = tmp_node2['x']
            y2 = tmp_node2['y']
            plt.arrow(x1,y1,x2-x1,y2-y1,color='w')
        num_nodes += 1
print(str(num_nodes) + " edges plotted on prob_map")
annot_prob_map_path = os.path.join(base_dir,"Example_prob_map_edges.png")
plt.savefig(annot_prob_map_path)
print("Annotated prob_map saved to file: " + annot_prob_map_path)

# Code for demonstrating the GAT and VGN parts are working properly
nfeat = 64
nclass = 1
dropout = 0.5
alpha = 0.5

# We now go through the nodes locations and take the features from concat_feat and put them into a new feature
# array which becomes part of the input to the GNN module
concat_feat = concat_feat.detach().numpy()
adj_matrix = nx.adjacency_matrix(graph).toarray()
n_vertices = graph.number_of_nodes()
# We assume that the image is square
n_vertices_per_dim = np.sqrt(n_vertices).astype(int)
# For the GAT layer, we create an input feature matrix of shape [n_vertices, nfeat]
gnn_feat = np.zeros((n_vertices, nfeat))
for i in range(len(nodelist)):
    tmp_node = nodelist[i]
    gnn_feat[i,:] = concat_feat[0,:,int(tmp_node['y']),int(tmp_node['x'])]

# Create GAT model and apply to the gnn_feat matrix and adjacency matrix
gat_model = GraphAttentionLayer(nfeat, 32, dropout, alpha)
gat_out, gat_prob = gat_model(torch.from_numpy(np.squeeze(gnn_feat)), torch.from_numpy(adj_matrix))
print("GAT applied to GNN features")
print("y shape: " + str(gat_out.shape))
print("prob shape: " + str(gat_prob.shape))

### The "tensorization" of gat_output ###
# change the shape of gat_output from [n_vertices, out_features] to [out_features, n_vertices]
# gat_out = torch.moveaxis(gat_out, 1, 0)
gat_out=gat_out.permute(1,0)
# reshape gat_output from [out_features, n_vertices] to
# [out_features, n_vertices_per_dim, n_vertices_per_dim]
gat_out = torch.reshape(gat_out, (32, n_vertices_per_dim, n_vertices_per_dim))
# add extra dimension to gat_output, final shape is [1, out_feat, n_vertices_per_dim, n_vertices_per_dim]
gat_out = torch.unsqueeze(gat_out, 0)

vgn_module = vgn_inference_module(nclass, nfeat, dropout, alpha, device, base_dir)
vgn_out = vgn_module(gat_out, cnn_feat)
print("VGN applied to GAT output and CNN intermediate features")
print("output shape: " + str(vgn_out.shape))
VGN_out_path = os.path.join(base_dir, "Example_VGN_out.png")
plt.close('all')
fig = plt.figure()
im = plt.imshow(np.squeeze(vgn_out.detach().numpy()))
fig.colorbar(im)
plt.savefig(VGN_out_path)
print("VGN output saved to file: " + VGN_out_path)
