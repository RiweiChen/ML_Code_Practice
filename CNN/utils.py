#!/usr/bin/env python
# -*- coding: utf-8 -*-



# utils.py - 定义一些实用函数：激活函数，激活函数的导数，随机初始化权重，等等。
import numpy as np

class logistic:
	'''
     定义逻辑回归函数
    '''
	def __init__(self, beta = 1.0):
		self.beta = beta
    #逻辑回归函数
	def func(self, x):
		return 2.0 / (1.0 + np.exp(-self.beta * x)) - 1.0
	
    #逻辑回归函数的导数
	def deriv(self, x):
		y = self.func(x)
		e = np.exp(-self.beta * x)
		return 2.0 * self.beta * x * e / ((1+e) ** 2.0)
	
	# 根据大小，来返回一个采样的范围
	def sampleInterval(self, prev, curr):
		d = (- 4.0) * np.sqrt(6.0 / (prev + curr))
		return [-d, d]

class tanh:
	def func(self, x):
		return 1.7159 * np.tanh(2.0 * x / 3.0)
	
	def deriv(self, x):
		t = np.tanh(2.0 * x / 3.0) ** 2.0
		return 1.144 * (1 - t)

	def sampleInterval(self, prev, curr):
		d = (- 1.0) * np.sqrt(6.0 / (prev + curr))
		return [-d, d]
