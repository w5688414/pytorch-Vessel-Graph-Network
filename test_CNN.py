import os
import VGG16
import torch
from dataset import Datasets
import torch.nn as nn
from torchvision.utils import save_image
from util import get_auc_ap_score,diceCoeff
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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
    data = r"./DRIVE/test"
    model_dir='cnn_result'
    train_data = Datasets(path=data)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)

    best_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16.Net(num_classes = 1).to(device)
    # model.load_state_dict('best_model.pth')
    PATH='best_model_Aug10.pth'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint)
 
    model.eval()
    all_labels = np.zeros((0,))
    all_preds = np.zeros((0,))
    dice_loss=[]
    for i,input_dict in enumerate(train_loader):
        images = input_dict['img']
        labels = input_dict['label']
        names=input_dict['img_name']
        images = images.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor).to(device)
        probability_map,cnn_feat,concat_feat = model(images)
        # print(images)
        # print(label)
        dice=diceCoeff(probability_map, labels, smooth=1, activation='sigmoid')
        dice_loss.append(dice.item())
        fg_prob_map=probability_map.cpu().detach().numpy()

        # labels = ((labels.astype(float)/255)>=0.5).astype(float)
        all_labels = np.concatenate((all_labels,np.reshape(labels.cpu().numpy(), (-1))))
        label_roi = labels.cpu().numpy()
        # print(label_roi.tolist())
        label_roi = (label_roi>=0.5).astype(float)
        label_roi=label_roi.reshape(-1).astype(float)
        fg_prob_map=fg_prob_map.reshape(-1)

        rnd_fpr, rnd_tpr, thresholds = roc_curve(label_roi, fg_prob_map)
        auc_score=auc(rnd_fpr, rnd_tpr)
        fg_prob_map=(fg_prob_map>=0.5).astype(float)
        auc_test, ap_test=get_auc_ap_score(label_roi,fg_prob_map)
        print('auc: {}'.format(auc_test))
        file_name=os.path.join('plot_cnn_results',names[0].split('.')[0]+'.png')
        draw_auc(rnd_fpr,rnd_tpr,roc_auc=auc_score,file_name=file_name)
        for i in range(images.shape[0]):
            y_pred = torch.squeeze(probability_map[i])
            y_pred = y_pred.detach().cpu().numpy().astype(np.float32)
            y = labels[i]
            save_image(torch.from_numpy(y_pred), os.path.join(model_dir, "{}_pred.png".format(names[i].split('.')[0])),normalize=True)
            save_image(y.detach().cpu(), os.path.join(model_dir, "{}_gt.png".format(names[i].split('.')[0])),normalize=True)
    print('mean dice codeff: {}'.format(np.mean(dice_loss)))