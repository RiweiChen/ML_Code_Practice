#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pooling.py - 实现下采样层的前馈和反馈操作

import numpy as np
from utils import *

class poolingConnection:
	'''
	下采样方法：这里采样的是MaxPooling 方法
	'''
	def __init__(self, prevLayer, currLayer, poolingStepX, poolingStepY):
		self.prevLayer   = prevLayer
		self.currLayer  = currLayer
		self.poolingStepX = poolingStepX
		self.poolingStepY = poolingStepY
		#校验连接层的大小是否匹配
		if self.prevLayer.shape()[0] / poolingStepY != self.currLayer.shape()[0] or \
		   self.prevLayer.shape()[1] / poolingStepX != self.currLayer.shape()[1]:
			raise Exception('Pooling step should match size ratio between consecutive layers')
		#下采样层不能改变特征图的个数
		if self.prevLayer.get_n() != self.currLayer.get_n():
			raise Exception('Number of feature maps before and after pooling should be the same')
	
	def propagate(self):
		'''
		前馈，取相对应的区域的最大值即可
		prev-->curr
		'''
		[prevSizeY, prevSizeX] = self.prevLayer.shape()
		[currSizeY, currSizeX] = self.currLayer.shape()
		#记录最大值的位置，以便于后面的反馈操作。
		self.maximaLocationsX = np.zeros([self.currLayer.get_n(), self.currLayer.shape()[0], self.currLayer.shape()[1]])
		self.maximaLocationsY = np.zeros([self.currLayer.get_n(), self.currLayer.shape()[0], self.currLayer.shape()[1]])
		#存储下采样后的特征图。
		pooledFM = np.zeros([self.currLayer.get_n(), self.currLayer.shape()[0], self.currLayer.shape()[1]])
		#yi:prev的网络权值
		yi = self.prevLayer.get_FM()

		for n in range(self.prevLayer.get_n()):
			for i in range(currSizeY):
				for j in range(currSizeX):
					#提取需要下采样的区域
					reg = yi[n, i*self.poolingStepY:(i+1)*self.poolingStepY, j*self.poolingStepX:(j+1)*self.poolingStepX]
					#获取下采样区域中的最大值的位置，是指代在特征图中的位置
					loc = np.unravel_index(reg.argmax(), reg.shape) + np.array([i*self.poolingStepY, j*self.poolingStepY])
					self.maximaLocationsY[n, i, j] = loc[0]
					self.maximaLocationsX[n, i, j] = loc[1]
					pooledFM[n, i, j] = yi[n, loc[0], loc[1]]
	
		self.currLayer.set_FM(pooledFM)

	def bprop(self):
		'''
		反馈：计算梯度的反馈
		curr -- > prev
		
		'''
		currErr = self.currLayer.get_FM_error()
		prevErr = np.zeros([self.prevLayer.get_n(), self.prevLayer.shape()[0], self.prevLayer.shape()[1]])
		
		[currSizeY, currSizeX] = self.currLayer.shape()

		for n in range(self.prevLayer.get_n()):
			for i in range(currSizeY):
				for j in range(currSizeX):
					#因为是MaxPooling 下采样，所以只有在最大值处才有反馈到，其它位置的梯度为0
					prevErr[n, self.maximaLocationsY[n, i, j], self.maximaLocationsX[n, i, j]] = currErr[n, i, j]


		self.prevLayer.set_FM_error(prevErr)


