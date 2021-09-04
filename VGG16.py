import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.branch_1_1 = torch.nn.Conv2d(3, 64, (3, 3),stride=(1,1),padding=(1,1))
        self.branch_1_2 = torch.nn.Conv2d(64, 64, (3, 3),stride=(1,1),padding=(1,1))

        self.branch_2_1 = torch.nn.Conv2d(64, 128, (3, 3),stride=(1,1), padding=(1,1))
        self.branch_2_2 = torch.nn.Conv2d(128, 128, (3, 3),stride=(1,1),padding=(1,1))
        

        self.branch_3_1 = torch.nn.Conv2d(128, 256, (3, 3),stride=(1,1),padding=(1,1))
        self.branch_3_2 = torch.nn.Conv2d(256, 256, (3, 3),stride=(1,1),padding=(1,1))
        self.branch_3_3 = torch.nn.Conv2d(256, 256, (3, 3),stride=(1,1),padding=(1,1))

        self.branch_4_1 = torch.nn.Conv2d(256, 512, (3, 3),stride=(1,1),padding=(1,1))
        self.branch_4_2 = torch.nn.Conv2d(512, 512, (3, 3),stride=(1,1),padding=(1,1))
        self.branch_4_3 = torch.nn.Conv2d(512, 512, (3, 3),stride=(1,1),padding=(1,1))

        self.dropout = torch.nn.Dropout(p=0.5,inplace=False)
        self.pooling = torch.nn.MaxPool2d(2, stride=2,padding=(0,0))
        num_ch = 16

        self.spe_1 = torch.nn.Conv2d(64, num_ch, (3, 3),stride=(1,1),padding=(1,1))
        self.spe_2 = torch.nn.Conv2d(128, num_ch, (3, 3),stride=(1,1),padding=(1,1))
        self.spe_3 = torch.nn.Conv2d(256, num_ch, (3, 3),stride=(1,1),padding=(1,1))
        self.spe_4 = torch.nn.Conv2d(512, num_ch, (3, 3),stride=(1,1),padding=(1,1))
        self.upsamp_2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamp_4 = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamp_8 = torch.nn.Upsample(scale_factor=8, mode='bilinear')

        self.output = torch.nn.Conv2d(num_ch * 4, num_classes, (1, 1),padding=(0, 0))

        self.cnn_feat = {}

    def forward(self, x):
        branch1 = self.branch_1_1(x) # 64
        branch1 = F.relu(branch1)
        branch1 = self.branch_1_2(branch1) #64
        branch1 = F.relu(branch1)
        # print(f"the size of branch1 is {branch1.shape}")
        part_1 = self.spe_1(branch1)
        part_1_dropout = self.dropout(part_1)
        # print(f"the size of part_1 is {part_1.shape}")
        branch2 = self.pooling(branch1)  # pooling

        branch2 = self.branch_2_1(branch2) # 128
        branch2 = F.relu(branch2)
        branch2 = self.branch_2_2(branch2) # 128
        branch2 = F.relu(branch2)
        # print(f"the size of branch2 is {branch2.shape}")

        part_2 = self.spe_2(branch2)
        part_2_dropout = self.dropout(part_2)
        part_2_up = self.upsamp_2(part_2)
        # print(f"the size of part_2 is {part_2.shape}")
        # print(f"the size of part_2_up is {part_2_up.shape}")

        branch3 = self.pooling(branch2) # pooling

        branch3 = self.branch_3_1(branch3) # 256
        branch3 = F.relu(branch3)
        branch3 = self.branch_3_2(branch3) # 256
        branch3 = F.relu(branch3)
        branch3 = self.branch_3_3(branch3) # 256
        branch3 = F.relu(branch3)
        # print(f"the size of branch3 is {branch3.shape}")

        part_3 = self.spe_3(branch3)
        part_3_dropout = self.dropout(part_3)
        part_3_up = self.upsamp_4(part_3_dropout)
        # print(f"the size of part3 is {part_3.shape}")
        # print(f"the size of part_3_up is {part_3_up.shape}")
        branch4 = self.pooling(branch3) # pooling

        branch4 = self.branch_4_1(branch4) # 512
        branch4 = F.relu(branch4)
        branch4 = self.branch_4_2(branch4) # 512
        branch4 = F.relu(branch4)
        branch4 = self.branch_4_3(branch4) # 512
        branch4 = F.relu(branch4)
        # print(f"the size of branch4 is {branch4.shape}")

        part_4 = self.spe_4(branch4)
        part_4_dropout = self.dropout(part_4)
        part_4_up = self.upsamp_8(part_4_dropout)
        # print(f"the size of part_4 is {part_4.shape}")
        # print(f"the size of part_4_up is {part_4_up.shape}")

        tensor_set = [part_1, part_2_up, part_3_up, part_4_up]
        spe_concat = torch.cat(tensor_set, dim=1)
        # print(spe_concat.shape)
        spe_concat_final = self.output(spe_concat)

        probability_map = F.sigmoid(spe_concat_final)

        self.cnn_feat[0] = part_1_dropout
        self.cnn_feat[1] = part_2_dropout
        self.cnn_feat[2] = part_3_dropout
        self.cnn_feat[3] = part_4_dropout
#        print(part_1_dropout.shape) torch.Size([4,16,592,592])
#        print(part_2_dropout.shape) torch.Size([4,16,296,296])
#        print(part_3_dropout.shape) torch.Size([4,16,148,148])
#        print(part_4_dropout.shape) torch.Size([4,16,74,74])

        return probability_map, self.cnn_feat, spe_concat

if __name__ == '__main__':
    a = torch.randn(1,3,592,592)
    net = Net(num_classes=1)
    print(net(a).shape)