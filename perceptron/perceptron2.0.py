# coding=UTF-8
# @Time: 2022/1/20 14:56
# @Author: 张 翔
# @File: perceptron2.0.py
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
训练集：10000个样本 784个特征  2种标签，感知机做了二值化处理（原本10种标签）

因为感知机只能一对多会出现一些难以辨别的区域
具体可参考邱锡鹏的《神经网络与深度学习》3.1.2 ：https://github.com/nndl/nndl.github.io
所以可以对标签做二值化分箱

初始化 w = [0]*n, b = 0

a=0.1, iters=100000:
模型用时：102.48
模型准确率：83.25%
"""

import pandas as pd
import numpy as np
from time import time


def data_load(train_filepath, test_filepath):
	"""
	:param train_filepath: 训练集路径
	:param test_filepath: 测试集路径
	:return: 训练集特征，测试集特征，训练集标签，测试集标签
	"""
	train_data = pd.read_csv(train_filepath, header=None)
	test_data = pd.read_csv(test_filepath, header=None)
	Xtrain, ytrain = train_data.iloc[:, 1:], train_data.iloc[:, 0]
	Xtest, ytest = test_data.iloc[:, 1:], test_data.iloc[:, 0]
	ytrain = np.where(ytrain >= 5, 1, -1)
	ytest = np.where(ytest >= 5, 1, -1)
	# Xtrain = Xtrain / 255           # 特征归一化用（Xtrain-最小值）/（最大值-最小值）
	# Xtest = Xtest / 255             # 用Xtrain.max().max()得知最大值为255，同理最小值为0
	# 试验过，归一化正确率上升一点
	return Xtrain, Xtest, ytrain, ytest


class Perceptron():
	def __init__(self, a=0.001, iters=50):
		"""
		:param a: 学习率
		:param iters:
		"""
		self.a = a
		self.iters = iters

	def fit(self, Xtrain, ytrain):
		"""
		:param Xtrain: 训练集特征
		:param ytrain: 训练集标签
		:return: 返回模型学得的参数w,b， 分割超平面为wx+b=0， 得到模型的参数
		有可能w=0但是b!=0，此时分割超平面不存在
		"""
		self.Xtrain = Xtrain
		self.ytrain = ytrain

		w = np.zeros(self.Xtrain.shape[1])
		b = 0
		tot_num = self.Xtrain.shape[0]
		i = 0
		while i < self.iters:
			for j in range(tot_num):  # 每个样本的特征
				x = self.Xtrain.iloc[j, :]
				y = self.ytrain[j]
				if y * (w.dot(x) + b) <= 0:
					w = w + self.a * y * x
					b = b + self.a * y
					i += 1
		self.w = w
		self.b = b

	def score(self, Xtest, ytest):
		"""
		:param Xtest: 测试集特征
		:param ytest: 测试集标签
		:return: 返回测试的正确率（正确样本数/总样本数）
		"""
		self.Xtest = Xtest
		self.ytest = ytest

		out = np.where((np.dot(self.Xtest, self.w) + self.b) < 0, -1, 1)  # 预测的标签列表
		return (self.ytest == out).sum() / self.ytest.shape[0]


def main():
	time0 = time()  # 时间标记
	Xtrain, Xtest, ytrain, ytest = data_load("../data/mnist_train.csv", "../data/mnist_test.csv")
	# 得到训练数据集和测试数据集
	perceptron = Perceptron(a=0.1, iters=100000)
	# 初始化模型 设置模型参数
	perceptron.fit(Xtrain, ytrain)
	# 训练模型
	score = perceptron.score(Xtest, ytest)
	# 用测试集打分
	time1 = time()  # 时间标记

	print(f"模型用时：{time1 - time0:.2f}\n"
		  f"模型准确率：{score:.2%}")


if __name__ == '__main__':
	main()
