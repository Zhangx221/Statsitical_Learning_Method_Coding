# coding=UTF-8
# @Time: 2022/1/23 22:34
# @Author: 张 翔
# @File: Logistic_regression2.0.py
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
测试集：50000个样本 784个特征   256种取值
训练集：10000个样本 784个特征  2种标签(1,0)与感知机的(1，-1)不同，做了二值化处理（原本10种标签）

所以可以对标签做二值化分箱

初始化 w = [0]*n + [1] # w和b合并处理

a=0.0001, iters=1000000得到：
模型用时：16.70
模型准确率：85.61%
"""

import pandas as pd
import numpy as np
from time import time


def data_load(train_filepath, test_filepath):
	"""
	:param train_filepath: 训练集路径
	:param test_filepath: 测试集路径
	:return: Xtrain, Xtest, ytrain, ytest ——> 训练特征，测试集特征，训练集标签，测试集标签
	四个返回值均为numpy数组格式
	"""
	train_data = pd.read_csv(train_filepath, header=None)
	test_data = pd.read_csv(test_filepath, header=None)
	Xtrain, ytrain = train_data.iloc[:, 1:], train_data.iloc[:, 0]
	Xtest, ytest = test_data.iloc[:, 1:], test_data.iloc[:, 0]
	ytrain = np.where(ytrain >= 5, 1, 0)  # 逻辑回归时候二分类，做二分箱处理，感知机为1和-1，这里为1和0
	ytest = np.where(ytest >= 5, 1, 0)
	Xtrain = np.array(Xtrain) / 255  # 没有归一化很容易在exp时候数值巨大，导致无法计算
	Xtest = np.array(Xtest) / 255
	return Xtrain, Xtest, ytrain, ytest


class Logistic_regression():
	def __init__(self, a=0.001, iters=10000):
		"""
		:param a:  学习率
		:param iters: 	迭代步数
		"""
		self.a = a
		self.iters = iters

	def fit(self, Xtrain, ytrain):
		Xtrain = np.hstack((Xtrain, np.ones(Xtrain.shape[0]).reshape(-1, 1)))  # 合并w和b也需要在x后面加一列1
		w = np.zeros(Xtrain.shape[1])  # 初始化w 为[0] * Xtrain加上1后的行数
		tot_num = Xtrain.shape[0]  # 总样本数
		i = 0  # w更新迭代次数初始化
		while i < self.iters:
			for j in range(tot_num):  # 每个样本的特征
				x = Xtrain[j]  # 某一个样本特征
				y = ytrain[j]  # 某一个样本标签
				expwx = np.exp(np.dot(w, x))  # w更新中的某一步，先计算了，为一个数
				w += self.a * (x * y - (expwx * x) / (1 + expwx))  # 等式后面为逻辑回归的对数似然函数，需要求其极大值，
				# 用反向的梯度下降法寻找极大值（w更新减号变成加法，反向更新）
				i += 1
		self.w = w

	def score(self, Xtest, ytest):
		"""
		:param Xtest: 测试集样本特征
		:param ytest: 测试集样本标签
		:return: 学习的模型的准确率
		"""
		Xtest = np.hstack((Xtest, np.ones(Xtest.shape[0]).reshape(-1, 1)))  # Xtest 后面也加一列1，方便与w运算,也可以将w拆成原始w和b
		p = np.exp(Xtest.dot(self.w)) / (1 + np.exp(Xtest.dot(self.w)))
		print(p)
		out = np.where(p < 0.5, 0, 1)  # 概率小于0.5输出0，大于0.5输出1
		return (ytest == out).sum() / ytest.shape[0]  # 分子为，判断准确的总和，分母为总测试样本数


def main():
	time0 = time()
	Xtrain, Xtest, ytrain, ytest = data_load("../data/mnist_train.csv", "../data/mnist_test.csv")
	lgr = Logistic_regression(a=0.0001, iters=1000000)
	lgr.fit(Xtrain, ytrain)
	lgr_score = lgr.score(Xtest, ytest)
	time1 = time()
	print(f"模型用时：{time1 - time0:.2f}\n"
		  f"模型准确率：{lgr_score:.2%}")


if __name__ == '__main__':
	main()
