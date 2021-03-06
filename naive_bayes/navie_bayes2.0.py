# coding=UTF-8
# @Time: 2022/1/23 22:21
# @Author: 张 翔
# @File: navie_bayes2.0.py
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
测试集：50000个样本 784个特征  256种取值
训练集：10000个样本 784个特征  10种标签

模型用时：56.25
模型准确率：83.45%
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
	Xtrain, Xtest = np.array(Xtrain), np.array(Xtest)
	return Xtrain, Xtest, ytrain, ytest


class Naive_bayes():
	"""
	:return: p(y): shape(10,)为label的10个取值
	以及p(x|y): shape(10,784,2) 为label的10个取值，对应的784个特征的01取值
	"""
	def __init__(self):
		self.py = []  # p(y)
		self.pyx = np.zeros((10, 784, 256))  # p(x|y)

	def fit(self, Xtrain, ytrain):

		for i in range(10):
			self.py.append(np.log((np.sum(ytrain == i) + 1) / (len(ytrain)+10)))  # 为什么对p(y)取log呢，一开始我没取，算出来概率一直为0，
		# 因为很多项在0~1之间的数相乘很容易趋近于0，达到python识别不到的的很小的数
		# 所以后面的概率都取了对数的形式，连成变成了连加，更好处理了

		for i in range(Xtrain.shape[0]):  # 在样本总数做循环
			y = ytrain[i]  # 当前样本标签
			x = Xtrain[i]  # 当前样本特征
			for j in range(784):
				self.pyx[y][j][x[j]] += 1
		# 记录p(x|y)在标签y下，相当于10个矩阵，每个矩阵784行（特征），256列（特征取值范围0~255）,元素对应每个x的取值的个数

		pyxn = np.zeros(256)
		for i in range(10):
			for j in range(784):  # 下面是对选定某个标签（y）下，某个特征（X的某一维度）下，256个值所在当前标签下总数的比例
				sums = 0  # 记录某个标签下，某个特征，的总记录个数
				for k in range(256):
					pyxn[k] = self.pyx[i][j][k]
					sums += self.pyx[i][j][k]
				for k in range(256):
					self.pyx[i][j][k] = np.log((pyxn[k] + 1) / (sums + 256))  # 拉普拉斯平滑，取lambda=1

	# 最后得到训练好的 p(y) : px, p(x|y) : pyx 保存在实例本身属性里面

	def score(self, Xtest, ytest):
		"""
		:param py:
		:param pyx:
		:param Xtest:
		:param ytest:
		:return:
		"""
		out = []
		for x in Xtest:
			prob = self.py  # 初始化概率=p(y)，因为这个每个统计学习方法公式4.7第一项就是y的概率
			for i, j in enumerate(x):
				prob += self.pyx[:, i, j]  # 对10个标签的，选定特征i，以及特征取值j的概率做连乘（因为是取了log所以变成了连加）
			out.append(np.argsort(prob)[-1])  # 取出10个标签中，对应y最大的索引（索引顺序刚好与标签顺序一直，所以索引值=标签值）
		return (out == ytest).sum() / ytest.shape[0]


def main():
	time0 = time()  # 时间标记
	Xtrain, Xtest, ytrain, ytest = data_load("../data/mnist_train.csv", "../data/mnist_test.csv")
	# 得到训练数据集和测试数据集
	nb = Naive_bayes()
	# 初始化模型
	nb.fit(Xtrain, ytrain)
	# 训练模型
	nb_score = nb.score(Xtest, ytest)
	# 用测试集打分
	time1 = time()  # 时间标记

	print(f"模型用时：{time1 - time0:.2f}\n"
		  f"模型准确率：{nb_score:.2%}")


if __name__ == '__main__':
	main()
