import random

class Layer(object):
    ''' 单个MLP的网络层 '''
    def __init__(self, num_neurons,
            weight_function=lambda: random.uniform(-0.2, 0.2)):
		#权重初始化方法，采用随机平均分布生成
        if num_neurons <= 0:
            raise ValueError

        self.num_neurons = num_neurons
        self.weight_function = weight_function #初始化函数
        self.next = None #前一层
        self.prev = None #后一层
        self.weights = None #两层网络之间的权重
        self.weight_changes = None #两层网络之间的权重跟新差值，保存已用来引入动量
        self.difs = None #损失值或者梯度
        self.has_bias = False #是否有偏置值
        self.values = [] #神经元的输入与输出值

    def next_layer(self, layer_instance):
        ''' 指定后一层 '''
        assert isinstance(layer_instance, Layer)
        self.next = layer_instance

    def prev_layer(self, layer_instance):
        ''' 指定前一层 '''
        assert isinstance(layer_instance, Layer)
        self.prev = layer_instance

    def init_values(self):
        ''' 初始化输入输出值 '''
        self.values = [0 for _ in range(self.num_neurons)]
        if self.has_bias:
            self.values[-1] = 1.

    def init_weights(self):
        '''初始化网络权重矩阵'''
        if self.next is not None:
            self.weights = []
            self.weight_changes = []
            for i in range(self.num_neurons):
                self.weights.append([self.weight_function()
                    for _ in range(self.next.num_neurons)])
                self.weight_changes.append([0
                    for _ in range(self.next.num_neurons)])

    def __str__(self):
        def prf(inp):
            return map(lambda x: '% .4f' % x, inp)

        out = '  V: %s\n' % prf(self.values)
        if self.weights:
            out += '  W: %s\n' % prf(self.weights)
        if self.weight_changes:
            out += '  C: %s\n' % prf(self.weight_changes)
        if self.difs:
            out += '  D: %s\n' % prf(self.difs)
        out += '\n'
        return out

