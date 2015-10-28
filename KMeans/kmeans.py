# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:35:19 2015

@author: PC-K4000-Win10
"""
import numpy as np
import math
import random
def loadDataSet(filename):
    '''
    加载数据，并保存在dataMat的数组中
    dataMat数组每一个元素为一个特征
    '''
    dataMat = []
    fid = open(filename,'r')
    for line in fid.readlines():
        featureItem = line.strip().split('\t')
        dataMat.append(map(float,featureItem))
    return dataMat
    
def euclideanDistance(feature1,feature2):
    '''
    计算两个向量的欧式距离
    '''
    return math.sqrt(sum(math.power(feature1-feature2,2)))   

def randomCenter(dataSet,k):
    '''
    数据集合初始化的K个聚类中心
    '''     
    n = np.shape(dataSet)[1]#数据的特征维度
    centers = np.mat(np.zeros((k,n)))
    for j in range(n):
        #保证初始化的数据中心都在原有的数据范围之内
        minValue = np.min(dataSet[:,j])
        maxValue = np.max(dataSet[:,j])
        rangeValue = maxValue - minValue
        centers[:,j] = minValue + rangeValue*random.rand(k,1)
    return centers
def KMeans(dataSet, k, distanceMeasure = euclideanDistance):
    m = np.shape(dataSet)[0]
    #用于存放每个点分配到的聚类中心
    clusterAssign = np.mat(np.zeros(m,2))
    centers = randomCenter(dataSet,k)
    clusterChange = True
    while clusterChange:
        clusterChange = False
        #计算每个点到已有的这K个点的距离，并将其分配给距离最近的中心
        for i in range(m):
            minDistance = np.inf
            minIndex = -1
            for j in range(k):
                currentDistance = distanceMeasure(dataSet[i,:],centers[j,:])
                if currentDistance < minDistance:
                    minDistance = currentDistance
                    minIndex = j
            if clusterAssign[i,0] != minIndex:
                clusterChange = True
            clusterAssign[i,:]=minIndex ,minDistance**2
        # 更新聚类中心
        for t in range(k):
            #找出分别属于每个类的数据点
            pt = dataSet[np.nonzero(clusterAssign[:,0].A == t)[0]]
            #用这些点作为新的数据中心
            centers[t,:]=np.mean(pt,axis = 0)
    return centers

def predict(data,centers,k):
    '''
    对于给定的数据data，根据离它距离最近的K个聚类中心
    '''
    minDistance= np.inf
    clusterAssign = -1
    for t in range(k):
        currentDistance = euclideanDistance(data,centers[t,:])
        if currentDistance < minDistance:
            minDistance = currentDistance
            clusterAssign = t
    return centers[clusterAssign]
    
