# coding=UTF-8
# @Time: 2022/1/17 10:52
# @Author: 张 翔
# @File: naive_bayes_2values.py
# @Software: PyCharm
# @email: 1456978852@qq.com

"""
#####################主机信息######################
Python implementation: CPython
Python version       : 3.8.8
IPython version      : 7.22.0

numpy     : 1.18.5
pandas    : 1.2.2

Compiler    : MSC v.1916 64 bit (AMD64)
OS          : Windows
Release     : 10
Machine     : AMD64
Processor   : Intel64 Family 6 Model 158 Stepping 13, GenuineIntel
CPU cores   : 8
Architecture: 64bit

#####################测试用例######################
测试集：50000个样本 784个特征
训练集：10000个样本 784个特征

分箱处理阈值为128，准确率与不分箱差不多，但是模型用时大幅度缩减
模型用时：33.311256408691406
模型准确率：84.37000%
"""

import numpy as np
import pandas as pd
from time import time


# 继续沿用以前的，只是做了二值化处理
def data_load(train_filepath, test_filepath):
	"""
	:param train_filepath: 训练集路径
	:param test_filepath: 测试集路径
	:return: 训练特征，测试集特征，训练集标签，测试集标签
	"""
	train_data, test_data = pd.read_csv(train_filepath, header=None), pd.read_csv(test_filepath, header=None)
	Xtrain, ytrain, Xtest, ytest = train_data.iloc[:, 1:], train_data.iloc[:, 0], test_data.iloc[:, 1:], test_data.iloc[
																										 :, 0]
	Xtrain = np.where(Xtrain >= 128, 1, 0)  # 分成黑白像素，因为不像KNN一养，朴素贝叶斯假设各个特征独立，
	# eg 尽量避免颜色为100的附近值可能为100左右这种潜在关系,进行分箱处理，虽然结果证明也差不多~ ~

	Xtest = np.where(Xtest >= 128, 1, 0)
	return Xtrain, Xtest, ytrain, ytest


def naive_bayes(Xtrain, ytrain):
	"""
	:param Xtrain:
	:param ytrain:
	:return: p(y): shape(10,)为label的10个取值
	以及p(x|y): shape(10,784,2) 为label的10个取值，对应的784个特征的01取值
	"""
	py = []  # p(y)
	pyx = np.zeros((10, 784, 2))  # p(x|y)
	for i in range(10):
		py.append((np.sum(ytrain == i)) / len(ytrain))

	for i in range(len(ytrain)):
		label = ytrain[i]
		x = Xtrain[i]
		for j in range(784):
			pyx[label][j][x[j]] += 1  # 对10个矩阵的每个特征为01（为二维的onehot向量）进行技术

	for i in range(10):
		for j in range(784):   # 这里下面两排是与不分箱的唯一差别 也可以将不分箱的k in range(2)
			pyx0, pyx1 = pyx[i][j][0], pyx[i][j][1]  # pyx0为 p(0|x,y)  # pyx1为 p(1|x,y)
			pyx[i][j][0], pyx[i][j][1] = np.log((pyx0 + 1) / (pyx0 + pyx1 + 2)), np.log((pyx1 + 1) / (pyx0 + pyx1 + 2))
	return py, pyx


def score(py, pyx, Xtest, ytest):
	"""
	:param py:
	:param pyx:
	:param Xtest:
	:param ytest:
	:return:
	"""
	outy = []
	for x in Xtest:
		prob = np.log(py)
		for i, j in enumerate(x):
			prob += pyx[:, i, j]
		outy.append(np.argsort(prob)[-1])
	return (outy == ytest).sum() / ytest.shape[0]


def main():
	time0 = time()
	Xtrain, Xtest, ytrain, ytest = data_load("../data/mnist_train.csv", "../data/mnist_test.csv")
	py, pyx = naive_bayes(Xtrain, ytrain)
	nb_score = score(py, pyx, Xtest, ytest)

	time1 = time()
	print(f"模型用时：{time1 - time0}\n"
		  f"模型准确率：{nb_score:.5%}")


if __name__ == '__main__':
	main()
