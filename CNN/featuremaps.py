#!/usr/bin/env python
# -*- coding: utf-8 -*-

# featuremaps.py - 定义二维的特征图

import numpy as np
from utils import *

class layerFM:
    '''
    
    n, width, height
    
    网络的输出表示：n个特征图，每个特征图大小为：width，height：即
    
    残差表示：n * width * height
    
    '''
    def __init__(self, n, width, height, isInput = False, isOutput = False):
		self.n = n
		self.width = width
		self.height = height
		self.FMs = np.zeros([n, width, height])
		self.error = np.zeros([n, width, height])
		self.isInput = isInput
		self.isOutput = isOutput
	
    def shape(self):
		return [self.height, self.width]	
	
    def get_n(self):
		return self.n

    def resetError(self):
		self.error = np.zeros([self.n, self.width, self.height])
	
    def addError(self, error):
		self.error += error
	
    def get_FM_error(self):
		return self.error
	
    def set_FM_error(self, error):
		self.error = error
	
    def set_FM(self, x):
	    if x.shape != self.FMs.shape: raise Exception("FeatureMap: set_x dimensions do not match")
	    self.FMs = x
	
    def get_FM(self):
		return self.FMs


	#特征图向量化
    def get_x(self):
		x = np.squeeze(self.FMs)
		if x.ndim != 1: raise Exception("Only 1x1 feature maps can be passed to a fully connected (1D) layer")
		return x
	
    def get_size(self):
		if self.height != 1 or self.width != 1: raise Exception("Only 1x1 feature maps can be connected to a fully connected (1D) layer")
		return self.n
	
    def set_error(self, err):
		self.error = err.reshape([len(err),1, 1,])
