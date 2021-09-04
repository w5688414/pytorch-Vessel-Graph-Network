from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import networkx as nx
from joblib.numpy_pickle_utils import xrange

def create_gnn_feats(concat_feat, graph, nfeat, delta=16):
    adj_matrix = nx.adjacency_matrix(graph)
    im_y = concat_feat.shape[-2]
    im_x = concat_feat.shape[-1]
    y_quan = range(0,im_y,delta)
    x_quan = range(0,im_x,delta)
    # For the GAT layer, we create an input feature matrix of shape [n_vertices, nfeat]
    gnn_feat = np.zeros((adj_matrix.shape[-1], nfeat))
    node_id = 0
    nodelist = graph.nodes
    for yi in xrange(len(y_quan)-1):
        for xi in xrange(len(x_quan)-1):
            tmp_node = nodelist[node_id]
            gnn_feat[node_id,:] = concat_feat[:,int(tmp_node['y']),int(tmp_node['x'])]
            node_id += 1
    return gnn_feat

class vgn_inference_module(nn.Module):
    def __init__(self, n_classes, nfeat, dropout, alpha, device, base_dir):
        super(vgn_inference_module, self).__init__()

        self.nfeat = nfeat
        temp_num_chs = 32
        self.conv1 = nn.Conv2d(temp_num_chs, 16, kernel_size=(1,1),stride=(1,1))
        self.conv2 = nn.Conv2d(32, 16, (3, 3),stride=(1,1), padding=(1,1))
        
        self.conv3= nn.Conv2d(32, 16, (3, 3),stride=(1,1), padding=(1,1))
        self.conv4 =nn.Conv2d(32, 16, (3, 3),stride=(1,1), padding=(1,1))

        self.upsamp_1 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamp_2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamp_3 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamp_4 = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        self.output = torch.nn.Conv2d(32, n_classes, (1, 1), padding=(0, 0))
  
    def forward(self, gat_output_feats, cnn_intermed_feats):
        # branch1
        post_cnn_conv_comp = self.conv1(gat_output_feats)
        current_input = F.relu(post_cnn_conv_comp)

        #print(post_cnn_conv_comp.shape)
        current_input=self.upsamp_1(current_input)
        #print(current_input.shape)
        #print(cnn_feat[0].shape)
        #print(cnn_feat[1].shape)
        #print(cnn_feat[2].shape)
        #print(cnn_feat[3].shape)

        # We concatenate first the fourth level of the CNN module with the current input
        current_input=torch.cat((current_input,cnn_intermed_feats[3]), 1)

        #print(current_input.shape)
        # branch2
        # current_input=self.conv1(current_input)
        current_input=self.conv2(current_input)
        current_input = F.relu(current_input)
        current_input=self.upsamp_2(current_input)
        current_input=torch.cat((current_input,cnn_intermed_feats[2]), 1)

        #print(current_input.shape)
        # branch 3
        current_input=self.conv3(current_input)
        current_input = F.relu(current_input)
        current_input=self.upsamp_3(current_input)
        current_input=torch.cat((current_input,cnn_intermed_feats[1]), 1)

        #print(current_input.shape)
        # branch 4
        current_input=self.conv4(current_input)
        current_input = F.relu(current_input)
        current_input=self.upsamp_4(current_input)
        current_input=torch.cat((current_input,cnn_intermed_feats[0]), 1)

        #print(current_input.shape)
        # final 1x1 convolution
        spe_concat_final = self.output(current_input)

        #final_result = F.sigmoid(spe_concat_final)
        final_result = spe_concat_final

        return final_result



