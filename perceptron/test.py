# coding=UTF-8
# @Time: 2022/1/12 23:06
# @Author: 张 翔
# @File: test.py
# @Software: PyCharm
"""
一个小的demo图形化表示感知机决策平面是怎么优化的
"""

import numpy as np
import matplotlib.pyplot as plt

x1 = (3, 3, 1)
x2 = (4, 3, 1)
x3 = (1, 1, -1)  # 一开始我把标签设置为0，发现到第二步就停止了，所以只能设为正负1才能使损失函数生效
data = [x1, x2, x3]

# 初始化w0和b
w = np.array((0, 0))  # 模型参数
w_list = []

b = 0  # 模型参数
b_list = []

a = 1  # 学习率
iters_nums = 0  # 迭代步数

max_round = 100  # 最大迭代步数<=此项*样本数

for i in range(max_round):

	for ith in data:
		if ith[2] * (ith[0] * w[0] + ith[1] * w[1] + b) <= 0:
			w = w + a * ith[2] * np.array(ith[0:2])
			b = b + a * ith[2]
			iters_nums += 1
			print(f"第{iters_nums}次迭代的点:{ith[0:2]}")
			print(f"w:({w[0]},{w[1]}),b:{b}\n")
			w_list.append(w)
			b_list.append(b)

for i in range(len(w_list)):
	plt.scatter(np.array(data)[:, 0], np.array(data)[:, 1], c=np.array(data)[:, 2])
	if w_list[i][0] == 0 and w_list[i][1] == 0:  # 分隔超平面不存在
		k, b = 0, 0
		plt.scatter(0, 0, label=f"第{i + 1}次迭代分割线")
	elif w_list[i][1] == 0:  # 斜率无穷大（不存在）
		plt.axvline(-b / w_list[i][0], label=f"第{i + 1}次迭代分割线")
	else:
		k = -w_list[i][0] / w_list[i][1]
		b = -b_list[i] / w_list[i][1]
		plt.plot((-2, 6), (k * (-2) + b, k * 6 + b), label=f"第{i + 1}次迭代分割线")
	plt.xlim((-2, 6))
	plt.ylim((-2, 6))
	plt.xlabel("w1")
	plt.ylabel("w2")
	plt.axvline(0, linestyle=":", c="black")
	plt.axhline(0, linestyle=":", c="black")
	plt.legend()
	plt.pause(0.5)
