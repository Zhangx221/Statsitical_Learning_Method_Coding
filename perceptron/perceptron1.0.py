# coding=utf-8
# @Time: 2022/1/13 18:28
# @Author: 张 翔
# @File: perceptron1.0.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from time import time
"""
#####################主机信息######################
Python implementation: CPython
Python version       : 3.8.8
IPython version      : 7.22.0

numpy     : 1.18.5
pandas    : 1.2.2
matplotlib: 3.3.3
scipy     : 1.6.2
sklearn   : 0.24.1

Compiler    : MSC v.1916 64 bit (AMD64)
OS          : Windows
Release     : 10
Machine     : AMD64
Processor   : Intel64 Family 6 Model 158 Stepping 13, GenuineIntel
CPU cores   : 8
Architecture: 64bit

#####################测试用例######################
取 a = 0.001 iters = 50得到：
模型用时：9.858847379684448
模型准确率：81.21000%

a=0.1, iters=100000:
模型用时：77.04006814956665
模型准确率：78.31000%
"""

def data_load(train_filepath, test_filepath):
    """
    :param train_filepath: 训练集路径
    :param test_filepath: 测试集路径
    :return: 训练特征，测试集特征，训练集标签，测试集标签
    """
    train_data = pd.read_csv(train_filepath, header=None)
    test_data = pd.read_csv(test_filepath, header=None)
    Xtrain, ytrain = train_data.iloc[:, 1:], train_data.iloc[:, 0]
    Xtest, ytest = test_data.iloc[:, 1:], test_data.iloc[:, 0]
    ytrain = np.where(ytrain >= 5, 1, -1)
    ytest = np.where(ytest >= 5, 1, -1)
    Xtrain = Xtrain / 255           # 特征归一化用（Xtrain-最小值）/（最大值-最小值）
    Xtest = Xtest / 255             # 用Xtrain.max().max()得知最大值为255，同理最小值为0
    return Xtrain, Xtest, ytrain, ytest


def perceptron(Xtrain, ytrain, a=0.001, iters=50):
    """
    :param Xtrain: 训练集特征
    :param ytrain: 训练集标签
    :param a: 学习率（learning rate，称为超参数）
    :param iters: iters为迭代次数
    :return: 返回模型学得的参数w,b， 分割超平面为wx+b=0， 有可能w=0但是b!=0，此时分割超平面不存在
    """
    w = np.zeros(Xtrain.shape[1])
    b = 0
    tot_num = Xtrain.shape[0]
    i = 0
    while i < iters:
        for j in range(tot_num):  # 每个样本的特征
            x = Xtrain.iloc[j, :]
            y = ytrain[j]
            if y * (w.dot(x) + b) <= 0:
                w = w + a * y * x
                b = b + a * y
                i += 1
    return w, b


def score(w, b, Xtest, ytest):
    """
    :param w: 模型参数
    :param b: 模型参数
    :param Xtest: 测试集特征
    :param ytest: 测试集标签
    :return: 返回测试的正确率（正确样本数/总样本数）
    """
    w = w.values
    tot_num = Xtest.shape[0]
    error_num = 0
    for j in range(tot_num):
        x = Xtest.iloc[j, :]
        y = ytest[j]
        if y * (w.dot(x) + b) <= 0:
            error_num += 1
    return 1 - error_num / tot_num


if __name__ == '__main__':
    time0 = time()
    Xtrain, Xtest, ytrain, ytest = data_load("../data/mnist_train.csv", "../data/mnist_test.csv")
    w, b = perceptron(Xtrain, ytrain, a=0.1, iters=100000)
    score = score(w, b, Xtest, ytest)
    time1 = time()
    print(f"模型用时：{time1-time0}\n"
          f"模型准确率：{score:.5%}")