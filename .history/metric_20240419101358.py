from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, f1_score
from sklearn.metrics.cluster._supervised import check_clusterings
from scipy import sparse as sp
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


def inference(loader, model, device, view, data_size, Z_mu):
    """
    :return:
    total_pred: prediction among all modalities
    pred_vectors: predictions of each modality, list
    labels_vector: true label
    Hs: high-level features
    Zs: low-level features
    """
    model.eval()
    z_all = []
    q_all = []
    x1_all = []
    x2_all = []
    z1_all = []
    z2_all = []
    labels_vector = []
    for step, (xs, y, idx) in enumerate(loader):
        z_mu = Z_mu[idx]
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            xrs, zs, qs, zs_var, mu, log_var, mus, log_vars = model.forward_all(xs, z_mu)
            z = sum(zs)/view
            z1_all.extend(zs[0].cpu().numpy())
            z2_all.extend(zs[1].cpu().numpy())
        # z_all.extend(z.cpu().detach().numpy())
        # pred = q_com.detach().cpu().numpy().argmax(axis=1)
        labels_vector.extend(y.numpy())
        x1_all.extend(xs[0].cpu().numpy())
        x2_all.extend(xs[1].cpu().numpy())
        # q_all.extend(pred)

    labels_vector = np.array(labels_vector).reshape(data_size)
    # q_all = np.array(q_all).reshape(data_size)
    return z_all, labels_vector, q_all, x1_all, x2_all, z1_all, z2_all

def valid(model, device, dataset, view, data_size, class_num, Z_mu): # , train_writer
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    z_all, labels_vector, q_all, x1_all, x2_all, z1_all, z2_all = inference(test_loader, model, device, view, data_size, Z_mu)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=class_num)  # n_clusters:number of cluster
    y_pred = kmeans.fit_predict(Z_mu.detach().cpu().numpy())
    # y_pred = kmeans.fit_predict(z1_all) # Z1
    nmi, ari, acc, pur = evaluate(labels_vector, y_pred)

    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))

    return nmi, ari, acc