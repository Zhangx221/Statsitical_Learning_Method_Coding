# coding=UTF-8
# @Time: 2022/1/24 20:58
# @Author: 张 翔
# @File: DecisionTree2.0.py
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
训练集：10000个样本 784个特征  做了二值化处理（原本10种标签）二叉树好分些

模型用时：299.75
模型准确率：84.43%
"""

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
	Xtrain, ytrain, Xtest, ytest = train_data.iloc[:, 1:], train_data.iloc[:, 0], test_data.iloc[:,
																				  1:], test_data.iloc[
																					   :, 0]
	Xtrain = np.where(Xtrain >= 128, 1, 0)
	Xtest = np.where(Xtest >= 128, 1, 0)
	ytrain = ytrain.values
	ytest = ytest.values
	return Xtrain, Xtest, ytrain, ytest


class DecisionTree():

	def __init__(self, epsilon=0.1):
		"""
		:param epsilon: 信息增益小于这个值，便会不再分支，将这个节点划分为叶节点
		"""
		self.epsilon = epsilon

	def f_HD(self, D_label):
		"""
		:param D_label: 对某个节点求其 H(D)
		:return:
		"""
		D_label = np.array(D_label).reshape(-1)
		HD = 0
		set_ytrain = set(D_label)
		for i in set_ytrain:
			pyi = np.sum(D_label == i) / D_label.shape[0]
			HD += -pyi * np.log2(pyi)
		return HD

	def f_HDA(self, Xtrain_selected_A, ytrain):
		"""
		:param Xtrain_selected_A: 特征A那一列特征
		:param ytrain: 对应的标签
		:return: 按照特征A划分的 H(D|A)
		"""
		HDA = 0
		Xtrain_selected_A = Xtrain_selected_A.reshape(-1)
		set_A_Xtrain = set(Xtrain_selected_A)
		D = Xtrain_selected_A.shape[0]
		for i in set_A_Xtrain:
			pAi = np.sum(Xtrain_selected_A == i) / D
			HDA += pAi * self.f_HD(ytrain[Xtrain_selected_A == i])
		return HDA

	def max_gDA(self, Xtrain, ytrain):  # 给出数据特征，以及标签，选择信息增益最大的标签，以及信息增益
		"""
		:param Xtrain:
		:param ytrain:
		:return: 信息增益最大的标签，以及信息增益
		"""
		Xtrain = np.array(Xtrain)
		ytrain = np.array(ytrain)
		init_gDA, select_feature = 0, 0
		Xtrain = np.array(Xtrain)
		HD = self.f_HD(ytrain)
		for feature in range(Xtrain.shape[1]):
			Xtrain_selected_feature = Xtrain[:, feature].reshape(-1)
			gDA = HD - self.f_HDA(Xtrain_selected_feature, ytrain)
			if gDA > init_gDA:
				init_gDA = gDA
				select_feature = feature
		return select_feature, init_gDA

	def leaf_node_class(self, leaf_node_ylable):
		"""
		:param leaf_node_ylable:  传入一维array
		:return:  得到这个一维数组上数量最多的标签
		"""
		node_class = {i: 0 for i in range(10)}
		for i in leaf_node_ylable:
			node_class[i] += 1
		node_class = pd.Series(node_class)
		max_index = np.argmax(node_class)
		return max_index

	def new_data(self, inputX, inputy, A, a):
		"""
		:param inputX:
		:param inputy:
		:param A: 选择一个特征A
		:param a: 剔除特征A中，等于某个值的所有行，相当于参考特征A进行分支
		:return: 按特征A，选定特征，分好的数据集和标签
		"""
		inputX = np.array(inputX)
		inputy = np.array(inputy).reshape(-1)
		outX = []
		outy = []
		for i in range(inputX.shape[0]):
			if inputX[i][A] == a:
				outX.append(np.hstack((inputX[i][0:A], inputX[i][A + 1:])))
				outy.append(inputy[i])
		return np.array(outX), np.array(outy)

	def decision_tree(self, *arg):
		"""
		:param arg:
		:param epsilon:
		:return:
		"""

		leaf_Xtrain, leaf_ytrain = arg[0][0], arg[0][1]
		leaf_y_dict = {i for i in leaf_ytrain}

		if len(leaf_y_dict) == 2:
			return leaf_ytrain[0]

		if leaf_Xtrain.shape[0] <= 0:
			return self.leaf_node_class(leaf_ytrain)

		Ag, e = self.max_gDA(leaf_Xtrain, leaf_ytrain)
		if e < self.epsilon:
			return self.leaf_node_class(leaf_ytrain)

		out_tree = {Ag: {}}
		out_tree[Ag][0] = self.decision_tree(self.new_data(leaf_Xtrain, leaf_ytrain, Ag, 0))
		out_tree[Ag][1] = self.decision_tree(self.new_data(leaf_Xtrain, leaf_ytrain, Ag, 1))
		return out_tree

	def fit(self, Xtrain, ytrain):

		self.tree = self.decision_tree((Xtrain, ytrain))


	def predict(self, Xtest_i, tree):
		"""
		:param Xtest_i: 一个样本
		:param tree: 建好的树结构（多层字典）
		:return:
		"""
		while True:
			# 假如tree为{222: {0: {333:6},1: {444:9}}}
			(key, value), = tree.items()
			# key为222
			# value为{0: {333: 6}, 1: {444: 9}}
			# 			选择特征222
			# 222特征为0	/       \ 222特征为1
			# 继续选择333判断   继续选择444判断
			# 		  /           \
			# 得到分类结果6          9
			if isinstance(tree[key], dict):
				# 如果当前节点（value）是字典类型则往下进行
				Xtest_i = list(Xtest_i)
				select_key = Xtest_i[key]  # 得到第222个特征的值
				del Xtest_i[key]
				# 选择这个key索引继续往下搜索
				# 以及用过的索引，在下一次搜索时，可以不用管了，删除222

				tree = value[select_key]  # 对222节点下面的树进行判断，并树向下进一级
				if not isinstance(tree, dict):  # 到达叶节点
					return int(tree)
			else:
				return int(value)

	def score(self, Xtest, ytest):
		"""
		:param Xtest:
		:param ytest:
		out: out为Xtest的预测结果
		:return:
		"""
		# tree =   # 预测测试集的
		out = np.zeros(Xtest.shape[0])
		for i in range(Xtest.shape[0]):
			out[i] = self.predict(Xtest[i], self.tree)

		return (ytest == out).sum() / ytest.shape[0]


def main():
	time0 = time()  # 时间标记
	Xtrain, Xtest, ytrain, ytest = data_load("../data/mnist_train.csv", "../data/mnist_test.csv")
	# 得到训练数据集和测试数据集
	DT = DecisionTree()
	# 初始化模型 设置模型参数
	DT.fit(Xtrain, ytrain)
	# DT.tree( Xtrain, ytrain)
	# 训练模型
	dt_score = DT.score(Xtest, ytest)
	# 用测试集打分
	time1 = time()  # 时间标记

	print(f"模型用时：{time1 - time0:.2f}\n"
		  f"模型准确率：{dt_score:.2%}")


if __name__ == '__main__':
	main()
