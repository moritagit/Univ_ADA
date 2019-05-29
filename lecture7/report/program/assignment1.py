# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def generate_data(sample_size=90, n_class=3):
    x = (
        np.random.normal(size=(sample_size//n_class, n_class))
        + np.linspace(-3., 3., n_class)
        ).flatten()
    y = np.broadcast_to(
        np.arange(n_class),
        (sample_size // n_class, n_class)
        ).flatten()
    return x, y


def train(x, y, h, lamb, n_class):
    n_sample = x.shape[0]
    theta = np.zeros((n_sample, n_class))
    K = np.exp(-(x - x[:, None])**2 / (2*h**2))
    for label in range(n_class):
        pi_y = (y == label).astype(int)
        theta[:, label] = np.linalg.inv(K.dot(K) + lamb*np.eye(len(K))).dot(K).dot(pi_y)
    return theta


def visualize(x, y, theta, h, num=100, path=None):
    X = np.linspace(-5, 5, num=num)
    K = np.exp(-(x - X[:, None]) ** 2 / (2 * h ** 2))
    logit = K.dot(theta)
    unnormalized_prob = np.exp(logit - np.max(logit, axis=1, keepdims=True))
    prob = unnormalized_prob / unnormalized_prob.sum(1, keepdims=True)

    plt.clf()
    plt.xlim(-5, 5)
    plt.ylim(-.3, 1.8)

    plt.plot(X, prob[:, 0], c='blue')
    plt.plot(X, prob[:, 1], c='red')
    plt.plot(X, prob[:, 2], c='green')

    plt.scatter(x[y == 0], -.1 * np.ones(len(x) // 3), c='blue', marker='o')
    plt.scatter(x[y == 1], -.2 * np.ones(len(x) // 3), c='red', marker='x')
    plt.scatter(x[y == 2], -.1 * np.ones(len(x) // 3), c='green', marker='v')

    if path:
        plt.savefig(path)
    plt.show()


def main():
    # settings
    n_sample = 90
    n_class = 3
    h = 2
    lamb =  1e-4
    fig_path = '../figures/assignment1_result.png'
    np.random.seed(0)

    # load data
    x, y = generate_data(n_sample, n_class)
    print(x.dtype)
    #print(x)
    #print(y)

    # train
    theta = train(x, y, h, lamb, n_class)

    # result
    print(f'#Sample: {n_sample}    #Class: {n_class}')
    print(f'h = {h}    lambda = {lamb}')
    print(f'theta: \n{theta}')
    visualize(x, y, theta, h, path=fig_path)


if __name__ == '__main__':
    main()
