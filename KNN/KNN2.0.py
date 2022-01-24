# coding=UTF-8
# @Time: 2022/1/23 21:58
# @Author: 张 翔
# @File: KNN2.0.py
# @Software: PyCharm
# @email: 1456978852@qq.com

'''
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
这个时间太慢了，少取点样本，可以再初始化模型那里修改
测试集：30000个样本 784个特征  256种取值
训练集：200个样本 784个特征 10种标签

K = 5
模型用时：188.79660439491272
模型准确率：95.50000%
'''

import numpy as np
import pandas as pd
from time import time


def data_load(train_filepath, test_filepath):
	"""
	:param train_filepath: 训练集路径
	:param test_filepath: 测试集路径
	:return: 训练特征，测试集特征，训练集标签，测试集标签
	"""
	train_data, test_data = pd.read_csv(train_filepath, header=None), pd.read_csv(test_filepath, header=None)
	Xtrain, ytrain, Xtest, ytest = train_data.iloc[:, 1:], train_data.iloc[:, 0], test_data.iloc[:, 1:], test_data.iloc[
																										 :, 0]
	return Xtrain, Xtest, ytrain, ytest


class KNN():
	def __init__(self, K=5, train_num=30000, test_num=200):
		"""
		:param K: 输入实例附近的K的点决定实例的标签
		:param train_num: # 训练样本数
		:param test_num: # 测试样本数
		"""
		self.K = K
		self.train_num = train_num
		self.test_num = test_num

	def norm2(self, x1, x2):
		"""
		:param x1: 向量1
		:param x2: 向量2
		:return: 两个向量的欧氏距离（范数2）
		"""
		return np.sqrt(np.sum(np.square(x1 - x2)))

	def fit(self, Xtrain, ytrain):
		"""
		:param Xtrain:
		:param ytrain:
		:return: 截取样本， 这个其实和__init__放在一起，但是为了和其他的保持一致，就加了一个这个
		"""
		self.Xtrain, self.ytrain = np.mat(Xtrain)[:self.train_num], ytrain[:self.train_num]  # 训练集只取了30000个样本

	def score(self, Xtest, ytest):
		"""
		:param Xtest:
		:param ytest:
		:return: 返回测试的正确率（正确样本数/总样本数）
		"""
		out = []  # 预测的标签列表
		Xtest = np.mat(Xtest)[:self.test_num]
		ytest = ytest[:self.test_num]

		normlist = np.zeros(self.Xtrain.shape[0])
		for i in Xtest:
			for k, j in enumerate(self.Xtrain):
				normlist[k] = self.norm2(i, j)  # normlist欧式距离列表
			labels = self.ytrain[np.argsort(normlist)[:self.K]]  # 取出最近五个数标签
			labellist = np.zeros(10)  # 初始化[0]*10标签计数
			for k in labels:
				labellist[k] += 1  # labels里面有最近的5个样本的标签，5个标签范围在0-9内，标签取值刚好对应labellist里面的索引位置
			max_out = np.argsort(labellist)[-1]  # 排序计数好的标签，取出出现最多的
			out.append(max_out)  # 确定Xtest每个样本的标签，添加到ylist
		out = np.array(out)
		ytest = np.array(ytest.values)
		return np.sum(out == ytest) / ytest.shape[0]  # 预测与实际相同返回True，在np.sum里，True=1,False=0,


# 自然np.sum(listy == ytest)得到预测对的总数量
# ytest.shape[0]为ytest的行数，也就是测试样本总数


def main():
	time0 = time()  # 时间标记
	Xtrain, Xtest, ytrain, ytest = data_load("../data/mnist_train.csv", "../data/mnist_test.csv")
	# 得到训练数据集和测试数据集
	knn = KNN(K=5, train_num=30000, test_num=200)
	# 初始化模型 设置模型参数
	knn.fit(Xtrain, ytrain)
	# 训练模型
	knn_score = knn.score(Xtest, ytest)
	# 用测试集打分
	time1 = time()  # 时间标记

	print(f"模型用时：{time1 - time0:.2f}\n"
		  f"模型准确率：{knn_score:.2%}")


if __name__ == '__main__':
	main()
