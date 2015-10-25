#!/usr/bin/env python
# -*- coding: utf-8 -*-

# conv.py - 实现卷积层的前馈与反馈


import numpy as np
from utils import *
from featuremaps import *

class convolutionalConnection:
	'''
	用卷积层来连接两个网络层
	'''
	def __init__(self, prevLayer, currLayer, connectionMatrix, kernelWidth, kernelHeight, stepX, stepY, useLogistic = False):
		self.connections  = connectionMatrix #用来指代每个卷积核所连接的上层特征图，为m*n：
		self.prevLayer   = prevLayer
		self.currLayer  = currLayer
		self.kernelHeight = kernelHeight
		self.kernelWidth  = kernelWidth
		self.stepX        = stepX # 用来提高感受域的作用，一般传统上设置为1.
		self.stepY        = stepY
		# 指代激活函数
		if useLogistic:
			self.act = logistic()
		else:
			self.act = tanh()

		if prevLayer.get_n()  != np.shape(self.connections)[0] or \
		   currLayer.get_n() != np.shape(self.connections)[1]:
		   	print "Connection matrix size = ", self.connections.shape
			print "first layer = ", self.prevLayer.get_n()
			print "second layer = ", self.currLayer.get_n()
		   	raise Exception("convolutionalConnection: connection matrix shape does not match number" \
			"of feature maps in connecting layers")

		if np.ceil((self.prevLayer.get_FM().shape[1] - self.kernelHeight) / self.stepY + 1) != self.currLayer.get_FM().shape[1] or \
		   np.ceil((self.prevLayer.get_FM().shape[2] - self.kernelWidth) / self.stepX + 1) != self.currLayer.get_FM().shape[2]:
		   	raise Exception("Feature maps size mismatch")

		# random init kernels
		self.nKernels = self.prevLayer.get_n() * self.currLayer.get_n()
	
		# compute number of units in each layer (required to initlize weights)
		nPrev = self.prevLayer.get_n() * self.prevLayer.get_FM().shape[1] * self.prevLayer.get_FM().shape[2] 
		nCurr = self.currLayer.get_n() * self.currLayer.get_FM().shape[1] * self.currLayer.get_FM().shape[2]

		# calculate interval for random weights initialization
		l, h = self.act.sampleInterval(nPrev, nCurr)
	
		# initialize kernels to random values
		nCombinations = self.prevLayer.get_n() * self.currLayer.get_n()
		self.k = np.random.uniform(low = l, high = h, size = [nCombinations, self.kernelHeight, self.kernelWidth])

		# initialize one bias per feature map
		self.biasWeights = np.random.uniform(low = l, high = h, size = [self.currLayer.get_n()])
	
	def propagate(self):
		'''
		前馈：根据上层的特征图和已经训练或初始化的滤波器，卷积形成下一层特征图
		'''
		FMs = np.zeros([self.currLayer.get_n(), self.currLayer.shape()[0], self.currLayer.shape()[1]])
		inFMs = self.prevLayer.get_FM()

		k = 0 # 卷积核索引，
		for j in range(self.currLayer.get_n()): # 每一个输出特征图
			for i in range(self.prevLayer.get_n()): # 每一个输入特征
				if self.connections[i, j] == 1:

					# 计算输出特征图的每一个特征
					for y_out in range(self.currLayer.shape()[0]):
						for x_out in range(self.currLayer.shape()[1]):

							# 输出特征图的每一个点，都是在其感受域内的卷积的结果(矩阵相乘的和，因为没有翻转卷积核，确切说是相关操作)
							for y_k in range(0, self.kernelHeight, self.stepY):
								for x_k in range(0, self.kernelWidth, self.stepX):
									FMs[j, y_out, x_out] += inFMs[i, y_out + y_k, x_out + x_k] * self.k[k, y_k, x_k]
							# 每个神经元都有一个bias
							FMs[j, y_out, x_out] += 1 * self.biasWeights[j]
				# 利用下一个卷积核
				k += 1

			# 最后脸上一个非线性激活函数
			FMs[j] = self.act.func(FMs[j])


		self.currLayer.set_FM(FMs)
		return FMs
	
	def bprop(self, ni, target = None, verbose = False):
		'''
		反馈操作：
		'''
		yi = self.prevLayer.get_FM() # 前层
		yj = self.currLayer.get_FM() # 后层

		# 获取当前层的损失函数值
		if not self.currLayer.isOutput:
			currErr = self.currLayer.get_FM_error()
		else:
			currErr = -(target - yj) * self.act.deriv(yj)
			self.currLayer.set_FM_error(currErr)

		# 计算前层的损失函数值
		prevErr = np.zeros([self.prevLayer.get_n(), self.prevLayer.shape()[0], self.prevLayer.shape()[1]])
		biasErr = np.zeros([self.currLayer.get_n()])

		k = 0 
		for j in range(self.currLayer.get_n()): # 遍历每一个后层特征图
			for i in range(self.prevLayer.get_n()): # 遍历每一个前层特征图
				if self.connections[i, j] == 1:

					#  遍历后层的一个特征图
					for y_out in range(self.currLayer.shape()[0]):
						for x_out in range(self.currLayer.shape()[1]):

							# 遍历每一个卷积核的感受域
							for y_k in range(0, self.kernelHeight, self.stepY):
								for x_k in range(0, self.kernelWidth, self.stepX):
									prevErr[i, y_out + y_k, x_out + x_k] += self.k[k, y_k, x_k] * currErr[j, y_out, x_out]

							# 加上偏置
							biasErr[j] += currErr[j, y_out, x_out] * self.k[k, y_k, x_k]
				# 下一个卷积核
				k += 1
		# 乘以 激活函数的导数，才是真正的梯度
		for i in range(self.prevLayer.get_n()):
			prevErr[i] = prevErr[i] * self.act.deriv(yi[i])

		for j in range(self.currLayer.get_n()):
			biasErr[j] = biasErr[j] * self.act.deriv(1)


		self.prevLayer.set_FM_error(prevErr)

		# 更新卷积核w
		dw = np.zeros(self.k.shape)
		dwBias = np.zeros(self.currLayer.get_n())
		k = 0 
		for j in range(self.currLayer.get_n()): # 遍历每一个后层特征图
			for i in range(self.prevLayer.get_n()): # 遍历每一个前层特征图
				if self.connections[i, j] == 1:

					# 遍历后层的一个特征图
					for y_out in range(self.currLayer.shape()[0]):
						for x_out in range(self.currLayer.shape()[1]):

							# 遍历每一个卷积核的感受域
							for y_k in range(0, self.kernelHeight, self.stepY):
								for x_k in range(0, self.kernelWidth, self.stepX):
									#
									dw[k, y_k, x_k] +=  yi[i, y_out + y_k, x_out + x_k] * currErr[j, y_out, x_out]

							# 计算偏置的梯度
							dwBias[j] += 1 * currErr[j, y_out, x_out]

				# 下一个卷积核
				k += 1


		#  更新卷积核
		self.k -= ni * d
		self.biasWeights -= ni * dwBias

