import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TCN import TemporalConvNet


def cal_linear_num(layer_num, num_timesteps_input):
    result = num_timesteps_input + 4 * (2**layer_num - 1)
    return result


class TCN(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            channel_size,
            layer_num,
            num_timesteps_input,
            kernel_size=3
    ):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=in_channels, num_channels=channel_size, kernel_size=kernel_size)
        linear_num = cal_linear_num(layer_num, num_timesteps_input)
        self.linear = nn.Linear(linear_num, out_channels)#
        # self.linear2 = nn.Linear(4, 3)

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)
        # if is_weather:
        #     if Weather.shape[0] != X.shape[0]:
        #         Weather = Weather.narrow(0, 0, X.shape[0])
        #     Weather = Weather.repeat(X.shape[1]*X.shape[2], 1).reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        #     Weather = torch.tensor(Weather)
        #     X = torch.cat((X, Weather), 3)
        #     X = self.linear2(X)
        X = self.tcn(X)
        X = self.linear(X)
        X = X.permute(0, 2, 1, 3)
        return X


def cal_channel_size(layers, timesteps_input):
    channel_size = []
    for i in range(layers - 1):
        channel_size.append(timesteps_input)
    channel_size.append(timesteps_input - 2)
    return channel_size


class DeepSTF(nn.Module):
    def __init__(
            self,
            num_nodes,
             out_channels,
             spatial_channels,
             features,
             timesteps_input,
             timesteps_output
    ):
        super(DeepSTF, self).__init__()
        self.spatial_channels = spatial_channels
        tcn_layer = 5
        channel_size = cal_channel_size(tcn_layer, timesteps_input)
        self.tcn1 = TCN(
                        in_channels=features,
                        out_channels=out_channels,
                        channel_size=channel_size,
                        layer_num=tcn_layer,
                        num_timesteps_input=timesteps_input
                    )
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        channel_size = cal_channel_size(tcn_layer, timesteps_input - 2)
        self.tcn2 = TCN(
                        in_channels=spatial_channels * 2,
                        out_channels=out_channels*8,
                        channel_size=channel_size,
                        layer_num=tcn_layer,
                        num_timesteps_input=timesteps_input - 2
                    )

        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()
        self.fully = nn.Linear(8*6, timesteps_output)
        self.conv2=nn.Conv2d(in_channels=10,out_channels=10,kernel_size=(3,3),padding=(1,1))
        self.matchconv = nn.Conv2d(in_channels=4, out_channels=spatial_channels*2, kernel_size=(3, 1), stride=1, bias=True)
        self.conv=nn.Conv2d(in_channels=num_nodes,out_channels=num_nodes,kernel_size=(1,3))
        # self.l=nn.Linear(2*12,timesteps_output)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, W_nodes, X, V_asist):#, Weather
        out=[]
        for i in range(X.shape[1]):
            j=X[:,i,:,:,:]
            V=V_asist[:,i,:,:,:]

            t = self.tcn1(j)
            lfs = torch.einsum("ij,jklm->kilm", [W_nodes, t.permute(1, 0, 2, 3)])
            t2 = F.relu(torch.matmul(lfs, self.Theta1))  # [5,29,3,16]

            V = V.permute((0, 3, 2, 1))
            V_asist_match = F.relu(self.matchconv(V)).permute((0, 3, 2, 1))
            gate_spatail = F.relu(torch.add(V_asist_match[:, :, :, 0:self.spatial_channels], t2))
            # Glu_out = torch.cat([gate_spatail, V_asist_match[:, :, :, -self.spatial_channels:]], dim=3)
            Glu_out = torch.cat([gate_spatail, V_asist_match[:, :, :, -self.spatial_channels:]], dim=3)
            Glu_out=F.relu(self.conv2(Glu_out.permute(0,2,1,3))).permute(0,2,1,3)

            t3 = self.tcn2(Glu_out)
            # t3=Glu_out
            t3=torch.sigmoid(self.conv(t3))

            out3 = self.batch_norm(t3)
            out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1))).reshape((out3.shape[0], 1, out3.shape[1], -1, 1))
            out.append(out4)
        a=torch.cat(out,dim=1)
        # a=self.l(a.reshape(a.shape[0],a.shape[1],-1))
        a=torch.sum(a,dim=1)
        return a


