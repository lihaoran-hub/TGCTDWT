import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device=torch.device("cuda:0")

def cal_linear_num(layer_num, num_timesteps_input):
    result = num_timesteps_input + 4 * (2**layer_num - 1)
    return result


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))


    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


def cal_channel_size(layers, timesteps_input):
    channel_size = []
    for i in range(layers - 1):
        channel_size.append(timesteps_input)
    channel_size.append(timesteps_input - 2)
    return channel_size



class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    # def __init__(self, in_channels, spatial_channels, out_channels_1, out_channels_2, num_nodes):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        # self.temporal1 = TCNBlock(in_channels=in_channels,
        #                            out_channels=out_channels)

        self.temporal1 = TimeBlock(in_channels=in_channels,
                                  out_channels=out_channels)

        self.Theta1 = nn.Parameter(torch.FloatTensor(25,
                                                     25))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)

        # self.temporal1 = TimeBlock(in_channels=in_channels,
        #                            out_channels=out_channels_1)
        # self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels_2,
        #                                              spatial_channels))
        # self.temporal2 = TimeBlock(in_channels=spatial_channels,
        #                            out_channels=out_channels_2)

        self.batch_norm = nn.BatchNorm1d(25)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        lfs = torch.einsum("ij,jk->ki", [A_hat, X.permute(1, 0)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        return self.batch_norm(t2)
        # return t3

class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, out_channels, spatial_channels, features, timesteps_input,
                 timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        # self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
        #                          spatial_channels=16, num_nodes=num_nodes)
        # self.block2 = STGCNBlock(in_channels=64, out_channels=64,
        #                          spatial_channels=16, num_nodes=num_nodes)

        # self.block1 = STGCNBlock(in_channels=num_features, out_channels_1=16, out_channels_2=64,
        #                          spatial_channels=16, num_nodes=num_nodes)
        # self.block2 = STGCNBlock(in_channels=64, out_channels_1=16, out_channels_2=64,
        #                          spatial_channels=16, num_nodes=num_nodes)

        # self.last_temporal = TimeBlock(in_channels=16, out_channels=64)
        # self.fully = nn.Linear(225,
        #                        num_timesteps_output)
        # self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64 * 2,
        #                        num_timesteps_output)

        self.lstm = torch.nn.LSTMCell(input_size=12, hidden_size=6)

        self.cnn=nn.Sequential(
            nn.Conv2d(
            in_channels=97,
            out_channels=97*2,
            kernel_size=(3,3),
            padding=(1,2)
        ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.cnn2=nn.Sequential(
            nn.Conv2d(
            in_channels=97,
            out_channels=97*2,
            kernel_size=(3,3),
            padding=(1,2)
        ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.f_conv=nn.Conv2d(
            in_channels=97,
            out_channels=97*2,
            kernel_size=(3,3),
            padding=(1,1)
        )

        self.f_conv2 = nn.Conv2d(
            in_channels=97*2,
            out_channels=97*2,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        self.f_conv3 = nn.Conv2d(
            in_channels=97*2,
            out_channels=97,
            kernel_size=(3, 3),
            padding=(1, 1)
        )




        self.lin = nn.Linear(36, 36)
        self.ln_f = nn.BatchNorm2d(97)
        self.lean=nn.Linear(6,6)
    def forward(self, A_hat, X, V_asist):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """


        out1 = torch.squeeze(X, 4).reshape(-1,97,12)
        state_h = torch.zeros(out1.shape[1], 6).to(device)
        state_c = torch.zeros(out1.shape[1], 6).to(device)
        al = []
        for i in range(out1.shape[0]):
            state_h, state_c = self.lstm(out1[i], (state_h, state_c))
            state_h = torch.tanh(state_h)
            state = torch.unsqueeze(state_h, dim=-1)
            al.append(state)
            state = torch.squeeze(state, dim=-1)
        # y = torch.autograd.Variable(torch.ones(32, 9 * 25), requires_grad=True).cuda()
        # out2 = torch.matmul(state, y).reshape(out1.shape[1], 25, 9)  #TODO 线性化替换
        al = torch.cat(al, dim=-1).permute(2, 0, 1)
        out2 = al.reshape(-1, 97, 6, 1)
        # f=torch.unsqueeze(out2[:, :, 0, :],dim=-1).permute(0,3,1,2)
        # f=self.cnn(out2)
        # f=self.cnn2(f)
        f=F.relu(self.f_conv(out2))#F.sigmoid(self.f_conv(out2))
        f=F.relu(self.f_conv2(f))
        f=F.relu(self.f_conv3(f))



        # out2[:, :, 0, :] = self.ln_f(out2[:, :, 0, :])
        # out2[:, :, 1, :] = self.ln_o(out2[:, :, 1, :])
        # out2[:, :, 2, :] = self.ln_s(out2[:, :, 2, :])
        # out2 = self.lin(F.relu(out2.reshape(-1, 36)))
        f=self.ln_f(f)


        # module = f #torch.cat([f], dim=1).permute(0,2,1,3)
        #out2 = out2.reshape(-1, 307, 3, 12)
        out2=self.lean(f.reshape(-1,6)).reshape(-1, 97, 6, 1)
        #out2 = state_h.reshape(-1, 307, 3, 12)
        return out2


