import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
from torch.utils.tensorboard import SummaryWriter

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import KMeans
# from scipy.optimize import linear_sum_assignment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# MNIST-USPS
# BDGP
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
Dataname = 'MNIST-USPS'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=1)
parser.add_argument("--temperature_l", default=1)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=1)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "MNIST-USPS":
    args.con_epochs = 500
    seed = 5

def z_KLD(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return KLD

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)


dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
print(model)
model = model.to(device)
Z_mu = torch.normal(mean=torch.zeros([data_size, args.feature_dim]), std=0.01).to(device)
Z_mu.requires_grad_(True)
optimizer1 = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
optimizer2 = torch.optim.Adam([Z_mu], lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

tot_loss = 0.
acc_best = 0.
nmi_best = 0.
ari_best = 0.
mes = torch.nn.MSELoss()
for epoch in range(1, 200):
    for batch_idx, (xs, _, idx) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer1.zero_grad()
        xrs, zs, mus, log_vars = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(mes(xrs[v], xs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer1.step()

for epoch in range(1, 400):
    total_loss = 0.
    for batch_idx, (xs, _, idx) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        z_mu = Z_mu[idx]
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        xrs, zs, qs, zs_var, mu, log_var, mus, log_vars = model.forward_all(xs, z_mu)
        q_zmu, _, _ = model.cluster_layer(z_mu)
        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_label(qs[v], qs[w]))

            loss_list.append(criterion.forward_label(qs[v], q_zmu))

            loss_list.append(mes(xrs[v], xs[v]))

            loss_list.append(mes(z_mu, zs[v].detach()))
        loss_list.append(0.2 * (z_KLD(mu, log_var)).mean())
        loss = sum(loss_list)
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        total_loss += loss.item()
    print(np.mean(total_loss/len(data_loader)))

    nmi, ari, acc = valid(model, device, dataset, view, data_size, class_num, Z_mu)
    if acc > acc_best:
        acc_best = acc
        nmi_best = nmi
        ari_best = ari
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(acc_best, nmi_best, ari_best))
