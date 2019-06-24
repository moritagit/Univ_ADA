# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def generate_data(n=200):
    x = np.linspace(0, np.pi, n // 2)
    u = np.stack([np.cos(x) + .5, -np.sin(x)], axis=1) * 10.
    u += np.random.normal(size=u.shape)
    v = np.stack([np.cos(x) - .5, np.sin(x)], axis=1) * 10.
    v += np.random.normal(size=v.shape)
    x = np.concatenate([u, v], axis=0)
    y = np.zeros(n)
    y[0] = 1
    y[-1] = -1
    return x, y


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


def calc_design_matrix(x, c, h, kernel):
    return kernel(x[None], c[:, None], h)


def lrls(x, y, h=1., l=1., nu=1., kernel=gauss_kernel):
    """

    :param x: data points
    :param y: labels of data points
    :param h: width parameter of the Gaussian kernel
    :param l: weight decay
    :param nu: Laplace regularization
    :return:
    """
    x_tilde = x[y!=0]
    K = calc_design_matrix(x, x, h, kernel)
    K_tilde = calc_design_matrix(x_tilde, x, h, kernel)

    W = K
    D = np.diag(W.sum(axis=1))
    L = D - W

    tmp = K_tilde.dot(K_tilde.T) + l*np.eye(len(K_tilde)) + 2*nu*K.dot(L).dot(K)
    theta = np.linalg.inv(tmp).dot(K_tilde).dot(y[y!=0])
    return theta


def supervised_train(x, y, h=1., l=1., kernel=gauss_kernel):
    K = calc_design_matrix(x, x, h, kernel)
    tmp = K.dot(K.T) + l*np.eye(len(K))
    theta = np.linalg.inv(tmp).dot(K).dot(y)
    return theta


def visualize(x, y, theta, h=1., grid_size=100, path=None):
    plt.xlim(-20., 20.)
    plt.ylim(-20., 20.)
    grid = np.linspace(-20., 20., grid_size)

    X, Y = np.meshgrid(grid, grid)
    mesh_grid = np.stack([np.ravel(X), np.ravel(Y)], axis=1)

    k = np.exp(
        -np.sum(
            (x.astype(np.float32)[:, None] - mesh_grid.astype(np.float32)[None])**2,
            axis=2).astype(np.float64)
            / (2 * h ** 2)
        )
    plt.contourf(
        X, Y,
        np.reshape(np.sign(k.T.dot(theta)), (grid_size, grid_size)),
        alpha=.4,
        cmap=plt.cm.coolwarm,
        )
    plt.scatter(x[y==0][:, 0], x[y == 0][:, 1], marker='$.$', c='black')
    plt.scatter(x[y==1][:, 0], x[y == 1][:, 1], marker='$X$', c='red')
    plt.scatter(x[y==-1][:, 0], x[y == -1][:, 1], marker='$O$', c='blue')
    if path:
        plt.savefig(path)
    plt.show()


def main():
    # settings
    h = 1.0
    lamb = 1.0
    nu = 1.0
    n = 200
    fig_path = '../figures/assignment1_result.png'
    np.random.seed(0)


    # generate data
    x, y = generate_data(n)
    #print(x.shape, y.shape)


    # train
    theta = lrls(x, y, h=h, l=lamb, nu=nu)

    # supervised
    #x, y = x[y!=0], y[y!=0]
    #theta = supervised_train(x, y, h=h, l=lamb,)


    # result
    print(f'#data: {n}')
    print(f'h = {h}  lambda = {lamb}  nu = {nu}')
    #print('theta = \n', theta)

    visualize(x, y, theta, path=fig_path)


if __name__ == '__main__':
    main()
