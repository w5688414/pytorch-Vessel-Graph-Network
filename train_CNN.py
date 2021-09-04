import os
import VGG16
import torch
from dataset import Datasets
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np

# base_dir = "C:\\Users" \
    #    "\\VGN_v5"
base_dir='.'
model_dir='cnn_result'
best_loss = float('inf')
epoch = 3000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16.Net(num_classes = 1).to(device)
criterion = nn.BCELoss()
data = os.path.join(base_dir, "DRIVE/training")

train_data = Datasets(path=data)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-03,eps=0.1)
total_step = len(train_loader)
for iter in range(epoch):
    for i, input_dict in enumerate(train_loader):
        images = input_dict['img']
        labels = input_dict['label']
        names=input_dict['img_name']
        loss_list = []
        images = images.type(torch.FloatTensor).to(device)
        label_new = labels.type(torch.FloatTensor).to(device)
        probability_map,cnn_feat,concat_feat = model(images)
        loss = criterion(probability_map,label_new)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x = images[0]
        if(iter%10==0):
            # Print out maximum probability for each image (for a working segmentation, the max probability
            # should be equal to 1)
            y_pred = torch.squeeze(probability_map[0])
            y_pred = y_pred.detach().cpu().numpy().astype(np.float32)
            print("Max of probability map for current image: " + str(np.max(y_pred)))
            y = label_new[0]
            # Save example image prediction and ground truth
            save_image(torch.from_numpy(y_pred), os.path.join(model_dir, f"{iter}_pred.png"),normalize=True)
            save_image(y.detach().cpu(), os.path.join(model_dir, f"{iter}_gt.png"),normalize=True)

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(deepcopy(model.state_dict()), r'./best_model.pth')

    print(f"Epoch: {iter}/{epoch}, Loss: {np.mean(loss_list)}")

