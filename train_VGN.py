from dataset import Datasets
import VGG16
from torch.utils.data import DataLoader
from model import vgn_inference_module, create_gnn_feats
from torch import nn
import torch
from torchvision.utils import save_image
import os
from GAT_tf1 import GraphAttentionLayer
import networkx as nx
import pickle
from joblib.numpy_pickle_utils import xrange
import numpy as np
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

def extract_vertex_gt_label(graph, gt_label):
    # Extract probabilities at vertices' locations, based on ground-truth label.
    #
    # Inputs:
    # graph [networkx.Graph]: Graph for the current image.
    # gt_label [numpy array]: Ground truth labels for image.
    im_y = concat_feat.shape[-2]
    im_x = concat_feat.shape[-1]
    y_quan = range(0,im_y,delta)
    y_quan = sorted(list(set(y_quan) | set([im_y])))
    x_quan = range(0,im_x,delta)
    x_quan = sorted(list(set(x_quan) | set([im_x])))
    gnn_node_prob = np.zeros((1, len(y_quan)-1,len(x_quan)-1))
    nodelist = graph.nodes
    node_id = 0
    for yi in xrange(len(y_quan)-1):
        for xi in xrange(len(x_quan)-1):
            tmp_node = nodelist[node_id]
            gnn_node_prob[:,yi,xi] = np.mean(gt_label[0,int(tmp_node['y']),int(tmp_node['x'])])
            node_id += 1

    return gnn_node_prob

if __name__=="__main__":
    delta = 16
    dist_thresh = 40
    # Base directory
    # base_dir = "C:\\Users"
    # '\\VGN_v5\\'
    base_dir = "."
    output_dir='result'
    data_path = os.path.join(base_dir,"DRIVE/training")
    graphs_path = os.path.join(data_path, "graphs_training_delta" + str(delta) +
                                     "_dist_thresh" + str(dist_thresh) + ".pickle")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    # Load the graphs that have been pre-created for the training set
    with open(graphs_path, 'rb') as f:
        # The graphs are saved into a dict object, with the key being the name of the image
        graphs_dict = pickle.load(f)

    train_data = Datasets(path=data_path)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

    # Training and model parameters
    dropout=0.5
    alpha=0.2
    epoch=3000
    n_classes=1
    nfeat=64

    # Load pre-trained CNN model
    cnn_model = VGG16.Net(num_classes = 1)
    best_model_path = os.path.join(base_dir,"best_model_Aug10.pth")
    cnn_model.load_state_dict(torch.load(best_model_path))
    # Optional freezing of CNN parameters by setting param.requires_grad = False in the below for loop
    for param in cnn_model.parameters():
        param.requires_grad = True
    cnn_model = cnn_model.to(device)
    print("Pre-trained CNN model loaded")
    cnn_criterion = nn.BCELoss()

    gat_model = GraphAttentionLayer(nfeat, 32, dropout, alpha).to(device)
    gat_criterion = nn.BCELoss()

    infer_model = vgn_inference_module(n_classes, nfeat, dropout, alpha, device, base_dir).to(device)
    # Use nn.BCELoss() if using a sigmoid before last layer of infer_model, otherwise use
    # nn.BCEWithLogitsLoss().
    infer_criterion = nn.BCEWithLogitsLoss()

    # We sum together the lists of the parameters of the CNN, GAT and inference modules,
    # see: https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603
    optimizer = torch.optim.Adam(list(cnn_model.parameters()) + list(gat_model.parameters())
        + list(infer_model.parameters()), lr=1e-3)
    best_loss = float('inf')
    writer = SummaryWriter("runs/vgn")
    for i in range(epoch):
        cnn_loss_list = []
        gat_loss_list = []
        infer_loss_list = []
        loss_list = []
        for i_batch, batch in enumerate(train_loader):
            # Get output from CNN, GAT and inference separately + calculate losses on each
            # sum losses and call .backward() on the summed loss
            # (see: https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603)
            img = batch['img'].type(torch.FloatTensor).to(device)
            name = batch['img_name']
            labels = batch['label'].type(torch.FloatTensor).to(device)

            # Get the CNN probability map and CNN features
            prob_map, cnn_interm_feat, concat_feat = cnn_model(img)
            cnn_loss = cnn_criterion(prob_map, labels)

            # Acquire the input features to the GNN using the graph and the concatenated feature matrix from the CNN
            tmp_graph = graphs_dict[name[0]]
            n_vertices = tmp_graph.number_of_nodes()
            # We assume that the number of vertices is equal in the y and x dimension
            n_vertices_per_dim = np.sqrt(n_vertices).astype(np.int)
            # GNN takes as input a tensor of size [batch_size, n_vertices, nfeat]
            gnn_feats = create_gnn_feats(concat_feat[0].detach().cpu().numpy(), tmp_graph, nfeat)
            gnn_feats = torch.from_numpy(gnn_feats).to(device)
            adj_mat = torch.from_numpy(nx.adjacency_matrix(tmp_graph).toarray()).to(device)

            # gat_output is the 32-feature feature matrix, gat_prob is the per-vertex probability from the GAT on
            # which we calculate the loss for the GAT
            gat_output, gat_prob = gat_model(gnn_feats, adj_mat)

            ### The "tensorization" of gat_output ###
            # first, change the shape of gat_output from [n_vertices, out_features]
            # to [out_features, n_vertices]
            # gat_output = torch.moveaxis(gat_output, 1, 0)
            gat_output=gat_output.permute(1,0)
            # then, reshape gat_output from [out_features, n_vertices] to
            # [out_features, n_vertices_per_dim, n_vertices_per_dim]
            gat_output = torch.reshape(gat_output, (32, n_vertices_per_dim, n_vertices_per_dim))
            # finally, add extra dimension to gat_output, final shape
            # is [1, out_feat, n_vertices_per_dim, n_vertices_per_dim]
            gat_output = torch.unsqueeze(gat_output, 0)

            # We extract the ground-truth vertex vessel/airway probability at each vertex location to calculate loss
            # of the GAT model
            node_gt_prob = extract_vertex_gt_label(tmp_graph, labels[0].detach().cpu().numpy())
            node_gt_prob = torch.from_numpy(node_gt_prob).float().to(device)
            gat_loss = gat_criterion(gat_prob, node_gt_prob)

            # Apply the inference module
            infer_out = infer_model(gat_output, cnn_interm_feat)
            infer_loss = infer_criterion(infer_out, labels)

            # Sum the losses from CNN, GAT, inference modules
            sum_loss = (cnn_loss + gat_loss + infer_loss)

            optimizer.zero_grad()
            # Perform the backwards propagation
            sum_loss.backward()
            optimizer.step()

            loss_list.append(sum_loss.item())
            cnn_loss_list.append(cnn_loss.item())
            gat_loss_list.append(gat_loss.item())
            infer_loss_list.append(infer_loss.item())

        writer.add_scalar("train_loss", np.mean(loss_list), i)

        if i % 10 == 0:
            print("Epoch %i of %i: loss = %.5f" % (i, epoch, np.mean(loss_list)))
            print("cnn_loss = %.5f, gat_loss = %.5f, infer_loss = %.5f" % (
                np.mean(cnn_loss_list), np.mean(gat_loss_list), np.mean(infer_loss_list)))
            save_image(infer_out.detach().cpu(),
                os.path.join(output_dir, f"ep{i}_{i_batch}_VGN_pred.png"), normalize=True)
            infer_thresh = infer_out.detach().cpu().numpy()
            infer_thresh[infer_thresh >= 0.5] = 1.
            infer_thresh[infer_thresh < 0.5] = 0.
            save_image(torch.from_numpy(infer_thresh),
                os.path.join(output_dir, f"ep{i}_{i_batch}_VGN_pred_thresh.png"), normalize=True)
            cnn_out_thresh = prob_map.detach().cpu().numpy()
            cnn_out_thresh[cnn_out_thresh >= 0.5] = 1.
            cnn_out_thresh[cnn_out_thresh < 0.5] = 0.
            save_image(torch.from_numpy(cnn_out_thresh),
                os.path.join(output_dir, f"ep{i}_{i_batch}_CNN_pred_thresh.png"), normalize=True)
            save_image(labels[0].detach().cpu(),
                os.path.join(output_dir, f"ep{i}_{i_batch}_VGN_gt.png"), normalize=True)
            save_image(gat_prob.detach().cpu(),
                os.path.join(output_dir, f"ep{i}_{i_batch}_GAT_pred.png"), normalize=True)
            save_image(node_gt_prob.detach().cpu(),
                os.path.join(output_dir, f"ep{i}_{i_batch}_GAT_gt.png"), normalize=True)
        loss=np.mean(loss_list)
        if loss < best_loss:
            print('loss from {} to {}'.format(best_loss,loss))
            best_loss = loss
            torch.save(deepcopy(gat_model.state_dict()), r'./checkpoints/best_gat_model.pth')
            torch.save(deepcopy(cnn_model.state_dict()), r'./checkpoints/best_cnn_model.pth')
            torch.save(deepcopy(infer_model.state_dict()), r'./checkpoints/best_infer_model.pth')
            


