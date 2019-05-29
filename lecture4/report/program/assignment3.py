# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def generate_sample(x_min=-3., x_max=3., sample_size=10):
    x = np.linspace(x_min, x_max, num=sample_size)
    y = x + np.random.normal(loc=0., scale=.2, size=sample_size)
    y[-1] = y[-2] = y[1] = -4  # outliers
    return x, y


def model(x, theta):
    f = theta[0] + theta[1] * x
    return f


def turkey_loss(r, eta):
    rho = (1 - (1 - (r/eta)**2)**3)/6
    rho[np.abs(r) > eta] = 1/6
    loss = (1/2) * np.sum(rho)
    return loss


def compute_loss(x, y, theta, eta):
    y_pred = model(x, theta)
    r = y_pred - y
    loss = turkey_loss(r, eta)
    return loss


def calc_Phi(x, theta):
    n = x.shape[0]
    b = theta.shape[0]
    phi = np.zeros((n, b))
    phi[:, 0] = theta[0]
    phi[:, 1] = theta[1] * x
    return phi


def calc_W(r, eta):
    w = (1 - (r/eta)**2)**2
    w[np.abs(r) > eta] = 0
    w = np.diag(w)
    return w


def update(theta, phi, y, w):
    theta = np.linalg.inv(phi.T.dot(w).dot(phi)).dot(phi.T).dot(w).dot(y)
    return theta


def solve(x, y, theta_initial, eta, eps=1e-4, n=5, max_iter=100):
    diffs = []
    theta = theta_initial
    for i in range(max_iter):
        theta_old = theta.copy()
        phi = calc_Phi(x, theta)
        y_pred = model(x, theta)
        r = y_pred - y
        w = calc_W(r, eta)
        theta = update(theta, phi, y, w)
        diff = np.linalg.norm(theta - theta_old)
        if len(diffs) < n:
            diffs.append(diff)
        else:
            if (max(diffs) - min(diffs)) < eps:
                break
            diffs = diffs[1:] + [diff]
    n_iter = i + 1
    return theta, n_iter


def main():
    #np.random.seed(0)  # set the random seed for reproducibility

    # create sample
    x_min, x_max = -3, 3
    sample_size = 50
    x, y = generate_sample(x_min, x_max, sample_size)
    # print(x.shape, y.shape)

    # hyper parameter
    eta = 1.5

    # parameter
    theta_init = np.random.rand(2)

    # solve
    theta, n_iter = solve(x, y, theta_init, eta, eps=1e-4, n=5, max_iter=100)

    # calc loss
    loss = compute_loss(x, y, theta, eta)

    # result
    print('eta: {}'.format(eta))
    print('theta_init: {}'.format(theta_init))
    print('theta: {}'.format(theta))
    print('loss: {:.4f}'.format(loss))
    print('n_iter: {}'.format(n_iter))

    # plot
    x_axis = np.linspace(x_min, x_max, 100)
    plt.scatter(x, y)
    plt.plot(x_axis, model(x_axis, theta))
    plt.savefig('../figures/assignment3_result_eta{}.png'.format(int(10*eta)))
    plt.show()


if __name__ == '__main__':
    main()
