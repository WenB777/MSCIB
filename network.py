import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class DDClustering(nn.Module):
    def __init__(self, inputDim, n_cluster):
        super(DDClustering, self).__init__()
        hidden_layers = [nn.Linear(inputDim, 256), nn.ReLU()]
        hidden_layers.append(nn.BatchNorm1d(num_features=256))
        self.hidden = nn.Sequential(*hidden_layers)
        self.withoutSoft = nn.Linear(256, n_cluster)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        hidden = self.hidden(x)
        withoutSoftMax = self.withoutSoft(hidden)
        output = self.output(withoutSoftMax)
        return output, withoutSoftMax, hidden


class FusionNet(nn.Module):
    def __init__(self, feature_dim):
        super(FusionNet, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )
        self.fc31 = nn.Linear(feature_dim, feature_dim)
        self.fc32 = nn.Linear(feature_dim, feature_dim)

    def forward(self, z):
        h = self.decoder(z)
        mu = self.fc31(h)
        log_var = torch.sigmoid(self.fc32(h))
        return mu, log_var


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

        self.fc21 = nn.Linear(feature_dim, feature_dim)
        self.fc22 = nn.Linear(feature_dim, feature_dim)

    def re_parametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # mul是乘法的意思，然后exp_是求e的次方并修改原数值  所有带"—"都是inplace的 意思就是操作后 原数也会改动

        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # 在cuda中生成一个std.size()的张量，标准正态分布采样，类型为FloatTensor
        else:
            eps = torch.FloatTensor(std.size()).normal_()  # 生成一个std.size()的张量，正态分布，类型为FloatTensor
        eps = Variable(eps)  # Variable是torch.autograd中很重要的类。它用来包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息。
        repar = eps.mul(std).add_(mu)
        return repar

    def forward(self, x):
        mu = self.fc21(self.encoder(x))
        log_var = torch.sigmoid(self.fc22(self.encoder(x)))
        z = self.re_parametrize(mu, log_var)
        return mu, log_var, z


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.view = view
        self.cluster_layer = DDClustering(feature_dim, class_num)
        self.fusion_layer = FusionNet(feature_dim)

    def forward(self, xs):
        xrs = []
        zs = []
        mus = []
        log_vars = []
        for v in range(self.view):
            x = xs[v]
            mu, log_var, z = self.encoders[v](x)
            xr = self.decoders[v](z)
            zs.append(mu)
            xrs.append(xr)
            mus.append(mu)
            log_vars.append(log_var)
        return xrs, zs, mus, log_vars

    def forward_all(self, xs, z_mu):
        xrs = []
        zs = []
        qs = []
        zs_var = []
        mus = []
        log_vars = []
        for v in range(self.view):
            x = xs[v]
            mu, log_var, z = self.encoders[v](x)
            q, _, _ = self.cluster_layer(mu)
            xr = self.decoders[v](z)
            qs.append(q)
            zs.append(mu)
            zs_var.append(log_var)
            xrs.append(xr)
            mus.append(mu)
            log_vars.append(log_var)
        # 信息瓶径
        mu, log_var = self.fusion_layer(z_mu)
        return xrs, zs, qs, zs_var, mu, log_var, mus, log_vars
