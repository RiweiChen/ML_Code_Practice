import numpy as np
import math
class LogisticRegression:
    def __init__(self):
        self.W = None
    def train(self,dataSet,labels):
        self.W = self.gradientAscent(dataSet,labels)
    def predict(self,data):
        return self.sigmoid(sum(self.W*data))
    def sigmoid(self,x):
    	'''
    	逻辑回归的判别函数
    	'''
    	return 1.0/(1.0+math.exp(-x))
    
    def gradientAscent(self,dataSet,labels):
    	'''
    	输入参数datas：训练数据矩阵，每一行为一个数据
    	输入参数labels：标签数据，为一个值。
    	要求参数数据匹配
    	'''
    	dataX = np.mat(dataSet)
    	#每列代表一个特征，每行代表不同的训练样本。
    	dataY = np.mat(labels).transpose()
    	#标签，将行向量转置为列向量
    	m,n = np.shape(dataX)
    	alpha = 0.001
    	#步长，也就是学习率
    	itera_num = 1000
    	#迭代次数
    	W = np.ones((n,1))
    	for i in range(itera_num):
    		H = self.sigmoid(dataX * W)
    		# H 是一个列向量，元素个数==m
    		error = dataY - H
    		W = W + alpha * dataX.transpose()*error
    	return W
    
    def stochasticGradientAscent(self,dataSet,labels):
        '''
        随机梯度上升法，避免每次更新参数都需要遍历整个数据集一边
        '''
        m,n = np.shape(dataSet)
        alpha = 0.01
        W = np.ones(n)
        for i in range(m):
            h = self.sigmoid(sum(W*dataSet[i]))
            error = labels[i] - h
            W = W + alpha*error*dataSet[i]
        return W
        
	