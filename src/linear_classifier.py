import numpy as np
from random import shuffle
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def hinge_loss(W, X, Y, reg):
    '''
    :param W:   权重矩阵
    :param X:   批输入
    :param Y:   标签
    :param reg: 正则参数
    :return:    损失和梯度
    '''
    loss = 0.0
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    n_samples = X.shape[0]

    for i in range(n_samples):
        scores = np.dot(X[i],W)
        correct_class_score = scores[int(Y[i])]
        for j in range(num_classes):
            if j == int(Y[i]):
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, int(Y[i])] -= X[i].T
    loss /= n_samples
    dW /= n_samples
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    return loss, dW

def cross_entropy_loss(W, X, Y, reg):
    '''
    :param W:   权重矩阵
    :param X:   批输入
    :param Y:   标签
    :param reg: 正则参数
    :return:    损失和梯度
    '''
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    n_samples = X.shape[0]

    for i in range(n_samples):
        scores = np.dot(X[i], W)
        shift_scores =  scores - max(scores)
        dom = np.log(np.sum(np.exp(shift_scores)))
        loss_i = -shift_scores[int(Y[i])] + dom
        loss += loss_i
        for j in range(num_classes):
            output = np.exp(shift_scores[j]) / sum(np.exp(shift_scores))
            if j == int(Y[i]):
                dW[:, j] += (-1 + output) * X[i].T
            else:
                dW[:, j] += output * X[i].T
        loss /= n_samples
        loss += reg * np.sum(W * W)
        dW = dW / n_samples + 2 * reg * W
    return loss, dW


        


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, x, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        '''
        :param x:   (N_samples, N_features)
        :param y:   (N_samples)
        :param learning_rate:   学习率
        :param reg: 正则参数
        :param num_iters:   训练迭代的次数
        :param batch_size:  每次迭代使用的样本数量
        :param verbose: 是否显示训练进度
        :return:
        保存每次迭代loss的列表
        '''
        n_samples, n_features = x.shape
        num_classes = int(np.max(y) + 1)
        if self.W is None:
            self.W = 0.001 * np.random.randn(n_features, num_classes)

        loss_history = []
        for i in range(num_iters):
            # 在每次迭代随机选择batch_size个数据
            mask = np.random.choice(n_samples, batch_size, replace=True)
            x_batch = x[mask]
            y_batch = y[mask]

            # 计算损失和梯度
            loss, grad = self.loss(x_batch, y_batch, reg)
            loss_history.append(loss)

            #更新参数
            self.W -= grad*learning_rate

            if verbose and i % 100 == 0:
                print('iteration %d / %d: loss %f' % (i, num_iters, loss))

        return loss_history

    def predict(self, x):
        y_pred = np.dot(x, self.W)
        y_pred = np.argmax(y_pred, axis=1)

        return y_pred

    def loss(self, x_batch, y_batch, reg):
        pass

class LinearHinge(LinearClassifier):

    def loss(self, x_batch, y_batch, reg):
        return hinge_loss(self.W, x_batch, y_batch, reg)

class LinearCrossEntropy(LinearClassifier):

    def loss(self, x_batch, y_batch, reg):
        return cross_entropy_loss(self.W, x_batch, y_batch, reg)

def load_data_set():
    trainData = []
    testData = []
    dfTrain = pd.read_csv('mnist_01_train.csv')
    dfTest = pd.read_csv('mnist_01_test.csv')
    indexlist = dfTrain.columns.tolist()
    trainLabel = dfTrain['label'].tolist()
    testLabel = dfTest['label'].tolist()
    for i in tqdm(range(len(trainLabel))):
        trainData.append([dfTrain[x][i] for x in indexlist])
    for i in tqdm(range(len(testLabel))):
        testData.append([dfTest[x][i] for x in indexlist])
    return np.array(trainData, dtype=float), np.array(testData,dtype=float), np.array(trainLabel,dtype=float), np.array(testLabel,dtype=float)

def plot_confusion_matrix(matrix):
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, ax=ax)
    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()

if __name__ == '__main__':
    trainData, testData, trainLabel, testLabel = load_data_set()
    meanimg = np.mean(trainData, axis=0)
    trainData -= meanimg
    testData -= meanimg

    trainData = np.hstack([trainData, np.ones((trainData.shape[0], 1))])
    testData = np.hstack([testData, np.ones((testData.shape[0], 1))])

    # 分别可使用两种线性分类模型
    classifier1 = LinearHinge()
    classifier2 = LinearCrossEntropy()

    classifier1.train(trainData, trainLabel, num_iters=2000, verbose=True, batch_size=100)
    y_pred = classifier1.predict(testData)
    score_svc = classification_report(testLabel, y_pred, digits=10)
    print(score_svc)
    plot_confusion_matrix(confusion_matrix(testLabel, y_pred, labels=[0, 1]))
