# coding=UTF-8
# @Time: 2022/1/16 12:51
# @Author: 张 翔
# @File: KNN.py
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


def norm2(x1, x2):
	"""
	:param x1: 向量1
	:param x2: 向量2
	:return: 两个向量的欧氏距离（范数2）
	"""
	return np.sqrt(np.sum(np.square(x1 - x2)))


def KNN(Xtrain, Xtest, ytrain, K=5):
	"""
	:param Xtrain:
	:param Xtest:
	:param ytrain:
	:param K: 最近临的K个实例决定样本的标签
	:return:
	"""
	listy = []
	Xtrain,ytrain = np.mat(Xtrain)[:30000],ytrain[:30000]  # 训练集只取了30000个样本
	Xtest = np.mat(Xtest)[:200]  # 测试集只取200个样本

	normlist = np.zeros(Xtrain.shape[0])
	for i in Xtest:
		for k, j in enumerate(Xtrain):
			normlist[k] = norm2(i, j)  # normlist欧式距离列表
		labels = ytrain[np.argsort(normlist)[:K]]  # 取出最近五个数标签
		labellist = np.zeros(10)  # 初始化[0]*10标签计数
		for k in labels:
			labellist[k] += 1  # labels里面有最近的5个样本的标签，5个标签范围在0-9内，标签取值刚好对应labellist里面的索引位置
		out = np.argsort(labellist)[-1]  # 排序计数好的标签，取出出现最多的
		listy.append(out)  # 确定Xtest每个样本的标签，添加到ylist
		# print(out)
	return listy


def score(listy, ytest):
	"""
	:param listy: 预测的标签列表
	:param ytest:
	:return:
	"""
	ytest = ytest[:200]
	listy = np.array(listy)
	ytest = np.array(ytest.values)
	return np.sum(listy == ytest) / ytest.shape[0]  # 预测与实际相同返回True，在np.sum里，True=1,False=0,
													# 自然np.sum(listy == ytest)得到预测对的总数量
													# ytest.shape[0]为ytest的行数，也就是测试样本总数

def main():
	time0 = time()
	Xtrain, Xtest, ytrain, ytest = data_load("../data/mnist_train.csv", "../data/mnist_test.csv")

	listy = KNN(Xtrain, Xtest, ytrain, K=5)

	Knn_score = score(listy, ytest)

	time1 = time()
	print(f"模型用时：{time1 - time0}\n"
		  f"模型准确率：{Knn_score:.5%}")


if __name__ == '__main__':
	main()