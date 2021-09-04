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
from util import get_auc_ap_score,diceCoeff
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

def draw_auc(fpr,tpr,roc_auc,file_name):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(file_name)


if __name__=="__main__":

    data = r"./data"
    model_path='checkpoints'
    model_dir='vgn_result'
    dropout=0.5
    alpha=0.2
    epoch=3000
    n_classes=1
    nfeat=64
    delta = 16
    dist_thresh = 40

    base_dir = "."
    split='test'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(base_dir,"DRIVE/{}".format(split))
    train_data = Datasets(path=data_path)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    # device = torch.device("cpu")
    graphs_path = os.path.join(data_path, "graphs_{}_delta".format(split) + str(delta) +
                                     "_dist_thresh" + str(dist_thresh) + ".pickle")
    with open(graphs_path, 'rb') as f:
        # The graphs are saved into a dict object, with the key being the name of the image
        graphs_dict = pickle.load(f)

    cnn_model = VGG16.Net(num_classes = 1)
    # best_model_path = os.path.join(base_dir,"best_model_Aug10.pth")
    best_model_path=os.path.join(model_path,'best_cnn_model.pth')
    cnn_model.load_state_dict(torch.load(best_model_path))

    cnn_model = cnn_model.to(device)
    print("Pre-trained CNN model loaded")

    gat_model = GraphAttentionLayer(nfeat, 32, dropout, alpha).to(device)
    gnn_path=os.path.join(model_path,'best_gat_model.pth')
    gat_model.load_state_dict(torch.load(gnn_path))
    print("Pre-trained GAT model loaded")
    
    infer_model = vgn_inference_module(n_classes, nfeat, dropout, alpha, device, base_dir).to(device)
    infer_path=os.path.join(model_path,'best_infer_model.pth')
    infer_model.load_state_dict(torch.load(infer_path))
    print("Pre-trained Infer model loaded")


    infer_model.eval()
    cnn_model.eval()
    gat_model.eval()
    dice_loss=[]
    for batch in train_loader:
        images = batch['img'].type(torch.FloatTensor).to(device)
        names = batch['img_name']
        labels = batch['label'].type(torch.FloatTensor).to(device)

        prob_map, cnn_interm_feat, concat_feat = cnn_model(images)

        tmp_graph = graphs_dict[names[0]]
        n_vertices = tmp_graph.number_of_nodes()
        # We assume that the number of vertices is equal in the y and x dimension
        n_vertices_per_dim = np.sqrt(n_vertices).astype(np.int)
        # GNN takes as input a tensor of size [batch_size, n_vertices, nfeat]
        gnn_feats = create_gnn_feats(concat_feat[0].detach().cpu().numpy(), tmp_graph, nfeat)
        gnn_feats = torch.from_numpy(gnn_feats).to(device)
        adj_mat = torch.from_numpy(nx.adjacency_matrix(tmp_graph).toarray()).to(device)

        gat_output, gat_prob = gat_model(gnn_feats, adj_mat)

        gat_output=gat_output.permute(1,0)
        gat_output = torch.reshape(gat_output, (32, n_vertices_per_dim, n_vertices_per_dim))
        gat_output = torch.unsqueeze(gat_output, 0)

        probability_map=infer_model(gat_output, cnn_interm_feat)

        dice=diceCoeff(probability_map, labels, smooth=1, activation='sigmoid')
        # print('dice: {}'.format(dice.item()))
        dice_loss.append(dice.item())
        fg_prob_map=probability_map.cpu().detach().numpy()
        # print(fg_prob_map.shape)

        # labels = ((labels.astype(float)/255)>=0.5).astype(float)
        label_roi = labels.cpu().numpy()
        # print(label_roi.tolist())
        label_roi = (label_roi>=0.5).astype(float)
        label_roi=label_roi.reshape(-1).astype(float)
        fg_prob_map=fg_prob_map.reshape(-1)

        rnd_fpr, rnd_tpr, thresholds = roc_curve(label_roi, fg_prob_map)
        auc_score=auc(rnd_fpr, rnd_tpr)

        fg_prob_map=(fg_prob_map>=0.5).astype(float)
        print(names)
        file_name=os.path.join('plot_results',names[0].split('.')[0]+'.png')
        draw_auc(rnd_fpr,rnd_tpr,roc_auc=auc_score,file_name=file_name)
        
        for i in range(images.shape[0]):
            y_pred = torch.squeeze(probability_map[i])
            y_pred = y_pred.detach().cpu().numpy().astype(np.float32)
            y = labels[i]
            save_image(torch.from_numpy(y_pred), os.path.join(model_dir, "{}_pred.png".format(names[i].split('.')[0])),normalize=True)
            save_image(y.detach().cpu(), os.path.join(model_dir, "{}_gt.png".format(names[i].split('.')[0])),normalize=True)
        # labels, preds
        auc_score=get_auc_ap_score(label_roi,fg_prob_map)
        # print(auc)
        print('auc: {}'.format(auc_score))
    print('mean dice codeff: {}'.format(np.mean(dice_loss)))
