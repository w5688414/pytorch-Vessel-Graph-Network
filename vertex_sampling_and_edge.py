import os
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import skfmm
# import torch
import skimage
from joblib.numpy_pickle_utils import xrange
import networkx as nx
# from torchvision.utils import save_image
from tqdm import tqdm 

def vgn_feature(prob_map,delta,geo_dist_thresh,temp_graph_path):
    seg_size_str = '%.2d_%.2d'%(delta,geo_dist_thresh)
    temp = (prob_map*255).astype(int)
    cur_save_path = os.path.join(temp_graph_path,'_prob.png')
    # temp1 = torch.from_numpy(temp)
    #save_image(temp1,cur_save_path)
    cur_save_gra_savepath = os.path.join(temp_graph_path,seg_size_str+'.graph_res')

    vesselness = prob_map

    # Note: the prob_map has shape [n_ch, n_feat, y, x], where n_ch=1 and n_feat=1 and y and x are the image
    # y and x coordinates, respectively
    im_y = vesselness.shape[-1]
    im_x = vesselness.shape[-2]
    y_quan = range(0,im_y,delta)
    y_quan = sorted(list(set(y_quan) | set([im_y])))
    x_quan = range(0,im_x,delta)
    x_quan = sorted(list(set(x_quan) | set([im_x])))

    max_val = []
    max_pos = []

    for yi in xrange(len(y_quan)-1):
        for xi in xrange(len(x_quan)-1):
            cur_patch = vesselness[y_quan[yi]:y_quan[yi+1],x_quan[xi]:x_quan[xi+1]]
            if np.sum(cur_patch)==0:
                max_val.append(0)
                max_pos.append((y_quan[yi]+cur_patch.shape[0]/2,x_quan[xi]+cur_patch.shape[1]/2))
            else:
                max_val.append(np.amax(cur_patch))
                temp = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                max_pos.append((y_quan[yi] + temp[0], x_quan[xi] + temp[1]))
    graph = nx.Graph()

    # add nodes
    nodeidlist=[]
    for node_idx, (node_y,node_x) in enumerate(max_pos):
        nodeidlist.append(node_idx)
        graph.add_node(node_idx,kind='MP',y=node_y,x=node_x,label=node_idx)
        # print('node label: '+ str(node_idx) + ', pos: ' + str(node_y) + ',' + str(node_x))

    speed = vesselness
    speed = numpy.squeeze(speed)
    speed[ speed < 0.00001 ] = 0.00001
    nodelist = graph.nodes
    for i,n in enumerate(nodelist):
        phi = np.ones_like(speed)
        phi[int(graph.nodes[i]['y']),int(graph.nodes[i]['x'])] = -1
        if speed[int(graph.nodes[i]['y']),int(graph.nodes[i]['x'])] == 0:
            continue
        neig = speed[max(0,int(graph.nodes[i]['y'])-1):min(yi,int(graph.nodes[i]['y'])+2),\
                    max(0,int(graph.nodes[i]['x'])-1):min(xi,int(graph.nodes[i]['x'])+2)]
        if np.mean(neig)<0.1:
            continue
        tt = skfmm.travel_time(phi,speed,narrow=geo_dist_thresh)
        for n_id in nodeidlist[i + 1:]:
            n_comp = nodelist[n_id]
            geo_dist = tt[int(n_comp['y']), int(n_comp['x'])]  # travel time
            if geo_dist < geo_dist_thresh:
                ### Let's not use the weight= option, to keep things simple ###
                graph.add_edge(n, n_id)
                #graph.add_edge(n, n_id, weight=geo_dist_thresh / (geo_dist_thresh + geo_dist))
    return graph

