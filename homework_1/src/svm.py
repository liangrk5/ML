import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
    return trainData, testData, trainLabel, testLabel


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
    kernellist = ['linear', 'rbf']
    Clist = [0.5, 1.0, 5.0]
    for c in Clist:
        for kernel in kernellist:
            svc = svm.SVC(C=c, kernel=kernel)
            svc.fit(trainData, trainLabel)
            y_pred = svc.predict(trainData)
            score_svc = classification_report(trainLabel, y_pred, digits=10)
            print(score_svc)
            plot_confusion_matrix(confusion_matrix(trainLabel, y_pred, labels=[0, 1]))





