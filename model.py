import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group 

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

# the overall module used for training
class Pct(nn.Module):
    def __init__(self, args, output_channels=40):
        # why do i need to pass pct as argument to super? what does this do?
        super(Pct, self).__init__()
        self.args = args
        # (in_channels, out_channels(num_filters))
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128) # 128 = 2 * 64
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):


        ############-------neighbor embedding-------############
        # x has dimension: (Batchsize, 3, 1024)
        # the conv1d may require a tranpose
        # These conv1d layers are actually equivalent to linear layers. Linear layer is a special case of conv layer, I guess
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()


        x = F.relu(self.bn1(self.conv1(x))) # linear 1
        x = F.relu(self.bn2(self.conv2(x))) # linear 2
        
        x = x.permute(0, 2, 1)
        # xyz.shape = (B, 1024, 3)
        # x.shape = (B, 1024, 64) = B, N, D

        # now x is the embedded version of points
        # xyz is the original coordinates

        # the following three lines form a Sample and Group layer
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)

        # feature [B, 512, 128] = B, N', D'

        # Sample and Group layer 2
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)
        # feature [B, 256, 256] = B, D'', N''

        ######-------end of neighbor embedding-------######


        x = self.pt_last(feature_1) # the Encoder


        x = torch.cat([x, feature_1], dim=1) # now x has shape B, 5 * D, N = B, 1280, 256

        # LBR (the green gaint block in the middle of graph)
        x = self.conv_fuse(x) # now x has shape B, D, N = B, 1024, 256

        # Maxpooling, should get rid of N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # now x has shape B, D = B, 1024
        

        # LBRD
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x) # x = [B, 512]

        # LBRD
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x) # x = [B, 256]


        # last linear
        x = self.linear3(x)

        return x # [B, 256]

class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        
        batch_size, _, N = x.size()

        # B, D, N
        # these 2 LBR layers are not seen in the paper?
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # self-attention layers
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

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
        # x_r = B, D, N
        x_r = torch.bmm(x_v, attention) # I have good reasons to believe the roles of q and k has been swapped
        # LBR
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x