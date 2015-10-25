def sigmoid(x):
	'''
	逻辑回归的判别函数
	'''
	return 1.0/(1.0+exp(-x))

def gradientAscent(datas,labels):
	'''
	输入参数datas：训练数据矩阵，每一行为一个数据
	输入参数labels：标签数据，为一个值。
	要求参数数据匹配
	'''
	dataX = mat(datas)
	#每列代表一个特征，每行代表不同的训练样本。
	dataY = mat(labels).transpose()
	#标签，将行向量转置为列向量
	m,n = shape(dataX)
	alpha = 0.001
	#步长，也就是学习率
	itera_num = 1000
	#迭代次数
	W = ones((n,1))
	for i in range(itera_num):
		H = sigmoid(dataX * W)
		# H 是一个列向量，元素个数==m
		error = dataY - H
		W = W + alpha * X.transpose()*error
	return W

def stochasticGradientAscent(datas,labels):
	