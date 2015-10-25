#!/usr/bin/env python
# -*- coding: utf-8 -*-

# mlp.py - implements a MultiLayer perceptron


import numpy as np
from utils import *

class layer1D:
	'''
	一维网络层，区别于卷积的2维网络层，作为全连接层、分类的输出层等。
	'''
	def __init__(self, n, isInput = False, isOutput = False, hasBias = None, x = None):
		self.n = n
		self.isInput = isInput
		self.isOutput = isOutput
		if isInput:
			self.withBias = False
		else:
			self.withBias = True
		if hasBias is not None:
			self.withBias = hasBias
		
		if x is None:
			self.x = np.zeros(n)
		else:
			if len(x) != self.n: raise Exception("Input (x) size should should be equal to n ("+str(n)+")")
			self.x = x
		
		self.b = None
		if hasBias:
			b = 1

		if isInput and isOutput: raise Exception("Neuron Layer can't be both an input layer and an output one")

	def set_x(self, x):
		if len(x) != self.n: raise Exception("Input (x) size should should be equal to n ("+str(n)+")")
		self.x = x

	def get_x(self):
		return self.x
	
	# returns number of neurons in layer
	def get_size(self):
		return self.n
	
	def hasBias(self):
		return self.withBias
	
	# compute MSE for an output layer
	def sampleMSE(self, expectedOutputs):
		if not self.isOutput: raise Exception("MSE should only be computed on output neurons")
		self.dErrors = np.sum( (self.x - expectedOutputs) ** 2.0) / 2.0
		return self.dErrors
	
	# set layer error
	def set_error(self, deriv):
		self.error = deriv

	# get layer error
	def get_error(self):
		return self.error


class fullConnection:
	'''
	全连接层：m*n的连接两个网络层
	激活函数默认选择：tanh(),也可以选择sigmoid
	'''
	def __init__(self, prevLayer, currLayer, useLogistic = False, w = None):
		self.prevLayer = prevLayer
		self.currLayer = currLayer

		if useLogistic:
			self.act = logistic()
		else:
			self.act = tanh()

		self.nPrev = prevLayer.get_size()
		self.nCurr = currLayer.get_size()

		l, h = self.act.sampleInterval(self.nPrev, self.nCurr)

		# if the current layer has a bias
		# we add another weight pointing to a permanent 1
		if currLayer.hasBias():
			self.nPrev += 1

		if w is None:
			self.w = np.random.uniform(low = l, high = h, size = [self.nPrev, self.nCurr])
		else:
			self.w = w

		return None
	
	def propagate(self):
		x = self.prevLayer.get_x()[np.newaxis]
		if self.currLayer.hasBias:
			x = np.append(x, [1])

		z = np.dot(self.w.T, x)
		
		# compute and store output
		y = self.act.func(z)
		self.currLayer.set_x(y)

		return y
	
	def bprop(self, ni, target = None, verbose = False):
		yj = self.currLayer.get_x()
		if verbose: 
			print "out = ", yj
			print "w = ", self.w

		# compute or retreive error of current layer
		if self.currLayer.isOutput:
			if target is None: raise Exception("bprop(): target values needed for output layer")
			currErr = -(target - yj) * self.act.deriv(yj)
			self.currLayer.set_error(currErr)
		else:
			currErr = self.currLayer.get_error()

		if verbose: print "currErr =  ", currErr

		yi = np.append(self.prevLayer.get_x(), [1])
		# compute error of previous layer
		if not self.prevLayer.isInput:
			prevErr = np.zeros(len(yi))
			for i in range(len(yi)):
				prevErr[i] = sum(currErr * self.w[i]) * self.act.deriv(yi[i])

			self.prevLayer.set_error(np.delete(prevErr,-1))

		# compute weight updates
		dw = np.dot(np.array(yi)[np.newaxis].T, np.array(currErr)[np.newaxis])
		self.w -= ni * dw
	
	def get_weights(self):
		return self.w
	
	def set_weights(self, w):
		self.w = w

