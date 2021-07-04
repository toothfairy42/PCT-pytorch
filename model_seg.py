import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group 
import math
import numpy as np


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        # in_channels = 128, out_channels = 128
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 128]) for the first time 
        
        x = x.permute(0, 1, 3, 2)  # [32, 512, 128, 32]
        
        x = x.reshape(-1, d, s) # this line merges B and N dimensions [16384, 128, 32], why?
        
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B * N, 128, 32
        x = F.relu(self.bn2(self.conv2(x))) # B * N, 128, 32
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # B * N, 128
        x = x.reshape(b, n, -1).permute(0, 2, 1) # regain dimensions B, N [B, N, 32, 128]
        return x
class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # x_q: B, N, D/4
        
        # x_k: B, D/4, N
        x_k = self.k_conv(x)
        
        # x_v: B, D, N
        x_v = self.v_conv(x)
        
        # energy: B, N, N
        energy = torch.bmm(x_q, x_k)

        # softmax + L1 norm
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # x_r.shape = B, D, N
        x_r = torch.bmm(x_v, attention) # query and key swapped?
        # LBR
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class PCTseg(nn.Module):
    def __init__(self, args, part_num=50):
        super(PCTseg, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.gather_local_0 = Local_op(in_channels=128, out_channels=64) # 128 = 2 * 64
        self.gather_local_1 = Local_op(in_channels=128, out_channels=128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

    def forward(self, x, cls_label):

        ############-------neighbor embedding-------############
        # x.shape = [B, 3, N]
        xyz = x.permute(0, 2, 1)
        batch_size, _, N = x.size()


        x = F.relu(self.bn1(self.conv1(x))) # LBR1
        x = F.relu(self.bn2(self.conv2(x))) # LBR2
        
        x = x.permute(0, 2, 1)
        # xyz.shape = [B, N, 3]
        # x.shape = [B, N, 64]

        # now x is the embedded version of points
        # xyz is the original coordinates

        # the following three lines form a Sample and Group layer
        new_xyz, new_feature = sample_and_group(npoint=N, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)

        # feature [B, 1024, 64] = B, N', D'

        # Sample and Group layer 2
        new_xyz, new_feature = sample_and_group(npoint=N, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)
        # feature_1 [B, 128, 1024] = B, D'', N''

        ######-------end of neighbor embedding-------######

        # self attention layers
        x1 = self.sa1(feature_1)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1) # [B, 4 * 128, N]


        # LBR
        x = self.conv_fuse(x) # [B, 1024, N] (local info)



        # Max-Average Pooling
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # reapeat label feature for each point

        # global feature concatenates max, mean and label information
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), dim=1)

        # integrate each point with global feature
        x = torch.cat((x, x_global_feature), dim=1) # x.shape = [B, 1024 * 3 + 64, N] (label conv outputs 64)

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        # x.shape = [B, num_parts, N]

        return x



if __name__ == '__main__':
    # testing
    model = PCTseg()
    x = torch.randn((42, 2048, 3))
    x = x.permute(0, 2, 1)
    label = torch.randn((42, 16))
    x = model(x, label)