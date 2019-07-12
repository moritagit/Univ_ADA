# -*- coding: utf-8 -*-


import numpy as np
import scipy.linalg

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_data(n=1000):
    a = 3. * np.pi * np.random.rand(n)
    x = np.stack(
        [a*np.cos(a), 30*np.random.random(n), a*np.sin(a)],
        axis=1)
    return a, x


def calc_norm(x, c, save_memory=False):
    if save_memory:
        n_x = x.shape[1]
        n_c = c.shape[0]
        d = x.shape[-1]
        norm = np.zeros((n_c, n_x))
        x = np.reshape(x, (n_x, d))
        c = np.reshape(c, (n_c, d))
        for i in range(len(x)):
            x_i = x[i]
            norm[i, :] = np.sum((x_i - c)**2, axis=-1)
    else:
        norm = np.sum((x - c) ** 2, axis=-1)
    return norm


def gauss_kernel(x, c, h, save_memory=False):
    norm = calc_norm(x, c, save_memory)
    ker = np.exp(- norm / (2*h**2))
    return ker


def calc_design_matrix(x, c, h, kernel, save_memory=False):
    return kernel(x[None], c[:, None], h, save_memory=save_memory)


def similarity_matrix(x, k=1):
    n = x.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        x_i = x[i, :]
        #W[i, i] = 1

        norms_i = np.sum((x_i - x)**2, axis=1)
        #indices = np.argsort(norms_i)
        #print(i, indices[0], indices[1])

        idx_nearest = np.argsort(norms_i)[1:1+k]
        W[i, idx_nearest] = 1
        W[idx_nearest, i] = 1

    #print(W)
    #print(np.where(W==1))
    #print(np.where(W != W.T))
    #print(W.sum(axis=1))
    return W


def train(x, d, eps=1e-8):
    W = similarity_matrix(x)
    #W = calc_design_matrix(x, x, 0.5, gauss_kernel, True)

    D = np.diag(W.sum(axis=1))
    L = D - W

    eigen_values, eigen_vectors = scipy.linalg.eig(L, D)

    '''
    indices = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[indices]
    eigen_vectors = eigen_vectors[indices]

    eigen_values[(-eps < eigen_values) & (eigen_values < eps)] = 0
    '''

    # normalize
    for i in range(len(eigen_vectors)):
        eigen_vectors[i] = eigen_vectors[i]/np.linalg.norm(eigen_vectors[i])

    #print(np.where(eigen_vectors.sum(axis=1) == 1))
    #print(eigen_values)
    #print(eigen_values[-1])  # 0
    #print(eigen_vectors[-1])  # 1
    #print((eigen_vectors.dot(D)*(eigen_vectors)).sum(axis=1))

    #eigen_vectors_reduced = eigen_vectors[(eigen_values>0), :][::-1]
    eigen_vectors_reduced = eigen_vectors[::-1][1:]

    Psi = eigen_vectors_reduced[:d].T
    return Psi


def visualize(x, z, a, path=None):
    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=a, marker='o')

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(z[:, 1], z[:, 0], c=a, marker='o')

    if path:
        plt.savefig(str(path))
    plt.show()


def main():
    # settings
    n = 1000
    n_components = 2
    fig_path = f'../figures/assignment2_result.png'
    np.random.seed(0)


    # generate data
    a, x = generate_data(n)
    #print(a.shape, x.shape)


    # preprocess
    mu = x.mean(axis=0)
    x = x - mu


    # train
    z = train(x, d=n_components)


    # result
    print(f'#Data: {n}')
    print(f'z shape: {z.shape}')
    print(f'z = \n{z}')

    visualize(x, z, a, path=fig_path)



if __name__ == '__main__':
    main()
