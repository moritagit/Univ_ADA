# -*- coding: utf-8 -*-


import pathlib
import numpy as np
import matplotlib.pyplot as plt


def load_data(n_label=None, n_train=None, n_test=None):
    data_dir = '../data/'
    data_dir = pathlib.Path(data_dir)
    categories = list(range(10))
    train_X = []
    test_X = []
    for category in categories[:n_label]:
        # train data
        data_path = data_dir / 'digit_train{}.csv'.format(category)
        data = np.loadtxt(str(data_path), delimiter=',')[:n_train]
        train_X.append(data)

        # test data
        data_path = data_dir / 'digit_test{}.csv'.format(category)
        data = np.loadtxt(str(data_path), delimiter=',')[:n_test]
        test_X.append(data)
    labels = categories[:n_label]
    return train_X, test_X, labels


def make_train_data(train_data, labels, label):
    train_X = []
    train_y = []
    for i in labels:
        data = train_data[i]
        train_X.extend(data)
        n_data = len(data)
        if i == label:
            train_y.extend([1] * n_data)
        else:
            train_y.extend([-1] * n_data)
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    return train_X, train_y


class GaussKernelModel(object):
    def __init__(self, n, bandwidth):
        self.n = n
        self.bandwidth = bandwidth
        self.theta = np.empty(n)

    def __call__(self, x, c):
        K = self.design_matrix(x, c)
        y_hat = K.T.dot(self.theta)
        return y_hat

    def kernel(self, x, c, save_memory=False):
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
        ker = np.exp(- norm / (2*self.bandwidth**2))
        return ker

    def design_matrix(self, x, c, save_memory=False):
        mat = self.kernel(x[None], c[:, None], save_memory)
        return mat

    def train(self, x, y, lamb=1e-4, save_memory=False):
        K = self.design_matrix(x, x, save_memory)
        self.theta = np.linalg.solve(
            K**2 + lamb*np.identity(len(K)),
            K.T.dot(y[:, None]),
            )
        return


def train_one_vs_others(train_data, labels, h, lamb=1e-4, save_memory=False):
    model = []
    print('Train: ', end='')
    for label in labels:
        print(f'{label}  ', end='')
        train_X, train_y = make_train_data(train_data, labels, label)
        n_data = len(train_y)
        model_i = GaussKernelModel(n_data, h)
        model_i.train(train_X, train_y, lamb, save_memory)
        model.append(model_i)
    print('\ndone\n')
    return model


def test(model, train_data, test_data, labels):
    n_label = len(labels)
    confusion_matrix = np.zeros((n_label, n_label), dtype=int)
    n_data_all = 0
    result = {}
    print('Test')
    for label in labels:
        print(f'Label: {label}\t', end='')

        # load
        test_X = test_data[label]
        n_data = len(test_X)
        n_data_all += n_data
        train_X = []
        for i in labels:
            data = train_data[i]
            train_X.extend(data)
        train_X = np.array(train_X)

        # predict
        preds = []
        for i in labels:
            pred = model[i](test_X, train_X).flatten()  # (n_data,)
            preds.append(pred)
        preds = np.array(preds).T  # (n_data, n_label)
        preds = np.argmax(preds, axis=1)  # (n_data,)

        # make confusion matrix
        for i in labels:
            n = (preds == i).sum()
            confusion_matrix[label, i] = n

        # calc accuracy
        n_correct = confusion_matrix[label, label]
        acc = n_correct / n_data
        print(f'#Data: {n_data}\t#Correct: {n_correct}\tAcc: {acc:.3f}')

        result[label] = {
            'data': n_data,
            'correct': n_correct,
            'accuracy': acc,
            }
    result['confusion_matrix'] = confusion_matrix

    # overall score
    n_crr_all = np.diag(confusion_matrix).sum()
    acc_all = n_crr_all / n_data_all
    result['all'] = {
        'data': n_data_all,
        'correct': n_crr_all,
        'accuracy': acc_all,
        }
    print(f'All\t#Data: {n_data_all}\t#Correct: {n_crr_all}\tAcc: {acc_all:.3f}')
    print()
    print('Confusion Matrix:\n', confusion_matrix)
    print()
    return result


def print_result_in_TeX_tabular_format(result):
    labels = list(range(10))
    print('Scores')
    for label in labels:
        print('{} & {} & {} & {:.3f} \\\\'.format(
            label,
            int(result[label]['data']),
            int(result[label]['correct']),
            result[label]['accuracy']
            ))
    print()
    print('Confusion Matrix')
    for i in labels:
        print('{}    '.format(i), end='')
        for j in labels:
            print(' & {}'.format(int(result['confusion_matrix'][i, j])), end='')
        print(' \\\\')
    return


def main():
    # settings
    bandwidth = 1.0
    lamb = 1e-4
    np.random.seed(0)

    print('Settings')
    print(f'bandwidth: {bandwidth}\tlambda (L2): {lamb}\n')

    # load data
    train_X, test_X, labels = load_data(n_label=10, n_train=None, n_test=None)

    # train
    model = train_one_vs_others(train_X, labels, bandwidth, lamb, save_memory=True)

    # test
    result = test(model, train_X, test_X, labels)
    print_result_in_TeX_tabular_format(result)

if __name__ == '__main__':
    main()
