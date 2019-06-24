# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_total, n_positive):
    x = np.random.normal(size=(n_total, 2))
    x[:n_positive, 0] -= 2
    x[n_positive:, 0] += 2
    x[:, 1] *= 2.
    y = np.empty(n_total, dtype=np.int64)
    y[:n_positive] = 0
    y[n_positive:] = 1
    return x, y


def calc_A(y, y_tilde, x_train, y_train):
    indices_i = np.where(y_train == y)[0]
    indices_i_tilde = np.where(y_train == y_tilde)[0]

    if len(indices_i) * len(indices_i_tilde) == 0:
        return 0

    x_tilde = x_train[indices_i_tilde]
    A = 0
    for idx_i in indices_i:
        A += np.sum(np.sqrt(np.sum((x_train[idx_i] - x_tilde)**2, axis=0)))
    A /= (len(indices_i) * len(indices_i_tilde))
    return A


def calc_b(y, x_train, y_train, x_test):
    x_i = x_test[y_train == y]

    if len(x_i) * len(x_test) == 0:
        return 0

    b = 0
    for x_i_dash in x_test:
        b += np.sum(np.sqrt(np.sum((x_i_dash - x_i)**2, axis=0)))
    b /= (len(x_i) * len(x_test))
    return b


def estimate_pi(x_train, y_train, x_test):
    x, y = x_train, y_train
    A_pp = calc_A(1, 1, x, y)
    A_pm = calc_A(1, -1, x, y)
    A_mm = calc_A(-1, -1, x, y)
    b_p = calc_b(1, x_train, y_train, x_test)
    b_m = calc_b(-1, x_train, y_train, x_test)

    pi_hat = (A_pm - A_mm - b_p + b_m) / (2*A_pm - A_pp - A_mm)
    pi_hat = min(1, max(0, pi_hat))
    return pi_hat


def cwls(train_x, train_y, test_x, is_weighted=True):
    n = train_y.shape[0]

    if is_weighted:
        pi_hat = estimate_pi(train_x, train_y, test_x)
        Pi = np.zeros(n)
        Pi[train_y == 1] = pi_hat
        Pi[train_y == -1] = 1 - pi_hat
        Pi = np.diag(Pi)
    else:
        Pi = np.eye(n)

    Phi = np.concatenate([np.ones(n)[:, np.newaxis], train_x], axis=1)

    theta = np.linalg.inv(Phi.T.dot(Pi).dot(Phi)).dot(Phi.T).dot(Pi).dot(train_y)
    return theta


def visualize(train_x, train_y, test_x, test_y, theta, is_weighted=True):
    str_weighted = 'weighted' if is_weighted else 'unweighted'
    for x, y, name in [(train_x, train_y, 'train'), (test_x, test_y, 'test')]:
        plt.xlim(-5., 5.)
        plt.ylim(-7., 7.)
        lin = np.array([-5., 5.])
        plt.plot(lin, -(theta[2] + lin * theta[0]) / theta[1])
        plt.scatter(x[y==0][:, 0], x[y==0][:, 1], marker='$O$', c='blue')
        plt.scatter(x[y==1][:, 0], x[y==1][:, 1], marker='$X$', c='red')
        plt.savefig('../figures/assignment3_result_{}_{}.png'.format(str_weighted, name))
        plt.show()


def main():
    # settings
    is_weighted = False
    np.random.seed(0)


    # generate data
    train_x, train_y = generate_data(n_total=100, n_positive=90)
    eval_x, eval_y = generate_data(n_total=100, n_positive=10)


    # train
    theta = cwls(train_x, train_y, eval_x, is_weighted=is_weighted)


    # result
    print('result')
    visualize(train_x, train_y, eval_x, eval_y, theta, is_weighted=is_weighted)


if __name__ == '__main__':
    main()
