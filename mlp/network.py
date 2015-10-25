import math
from layer import Layer

class MLP(object):
    ''' 多层感知机实现 '''
    def __init__(self, activation_fn=lambda x: math.tanh(x),
            derivative_fn=lambda x: 1.0 - x**2 ):
		#激活函数及其激活函数的导函数
		#当激活函数为tanh()时，其导函数为1-tanh()^2
		#当激活函数为sigmoid()时,其导函数为：sigmoid()*(1-sigmoid())
        self.activation_fn = activation_fn
        self.derivative_fn = derivative_fn
        self.layers = []
		#学习率
        self.step = 0.3
		#动量
        self.moment = 0.1
        self.verbose = False

    def add_layer(self, layer_instance):
        ''' 往MLP中添加网络层，放在网络层数组的最后面		'''
        assert isinstance(layer_instance, Layer)
        self.layers.append(layer_instance)

    def init_network(self):
        ''' 设置每一层之间的连接关系，并初始化网络权重 '''
        for i in range(len(self.layers)-1):
            self.layers[i].next_layer(self.layers[i+1])
            self.layers[i].init_weights()

        for i in range(1, len(self.layers)):
            self.layers[i].prev_layer(self.layers[i-1])

        for lay in self.layers:
            lay.init_values()

    def add_bias(self):
        '''对输入层，添加一个偏置值，相应的神经元数目也加1'''
        zero_layer = self.layers[0]
        zero_layer.num_neurons += 1
        zero_layer.has_bias = True

    def train(self, patterns, target_error=0.01, max_iterations=10000):
        ''' 
		训练网络
		训练目标：错误率已经足够的低，或者迭代次数已经到达一定的阈值
		'''
        iters = 0
        while True:
            error = 0.
            iters += 1
            for input, target in patterns:
                self.forward(input)
                error += self.backward(target)
            if self.verbose:
                print '%d %.8f' % ( iters, error )
            if error < target_error or iters == max_iterations:
                break
        return (error, iters)

    def forward(self, input):
        ''' 前馈一次，'''
        zero_layer = self.layers[0]
        required = zero_layer.num_neurons
        if zero_layer.has_bias:
            required -= 1
        if required != len(input):
            raise ValueError
		# 输入赋值给神经元	
        zero_layer = self.layers[0]
        if zero_layer.has_bias:
            for i in range(zero_layer.num_neurons-1):
                zero_layer.values[i] = input[i]
        else:
            zero_layer.values = input
        #前馈传播
		for layer in self.layers[1:]:
            lim = layer.num_neurons
            if layer.has_bias:
                lim -= 1
            for idx in range(lim):
                val = .0
                for h_idx, h_neuron_value in enumerate(layer.prev.values):
                    val = val + h_neuron_value * layer.prev.weights[h_idx][idx]
                layer.values[idx] = self.activation_fn(val)
		# 返回最后一层的输出
        return self.layers[-1].values


    def backward(self, desired):
        ''' 反馈传播 '''
        difs = []
        total_error = 0.
		#计算梯度
        for layer in reversed(self.layers):
            layer.difs = []
            if layer.next is None:
			#输出层的损失函数，直接MSE
                for idx, value in enumerate(layer.values):
                    err = desired[idx] - value
                    total_error = (err**2)/2
                    layer.difs.append(err)
            else:
                for idx, value in enumerate(layer.values):
                    dif = 0.
                    err = 0.
                    for l_idx, l_dif in enumerate(layer.next.difs):
                        err += l_dif * layer.weights[idx][l_idx]

                    dif = self.derivative_fn(value) * err
                    layer.difs.append(dif)
		#更新权重
        for layer in self.layers:
            if layer.next is None:
                continue
            for i in range(layer.num_neurons):
                for j in range(layer.next.num_neurons):
                    weight_change = layer.values[i] * layer.next.difs[j]
                    layer.weights[i][j] += self.step * weight_change + \
                        self.moment * layer.weight_changes[i][j]
                    layer.weight_changes[i][j] = weight_change

        return total_error

    def __str__(self):
        out = 'MLP:\n'
        for layer in self.layers:
            out += '%s' % layer
        return out

    def as_graph(self):
        ''' Output dot graph representation '''
        out = 'digraph mlp { '
        for layer_id, layer in enumerate(self.layers):
            out += 'subgraph cluster_%d {' % layer_id
            for neu_id, value in enumerate(layer.values):
                out+= 'N%d_%d [label="N%d_%d=%.2f"];' % (
                    layer_id, neu_id, layer_id, neu_id, value)
            out += '}'
        for layer_id, layer in enumerate(self.layers):
            if layer.next is None:
                break
            for neu_id in range(layer.num_neurons):
                for next_id in range(layer.next.num_neurons):
                    label = ''
                    if layer.weights:
                        label = '[label="%.2f"]' % (
                            layer.weights[neu_id][next_id],)
                    out += 'N%d_%d -> N%d_%d %s;' % (layer_id, neu_id,
                        layer_id+1, next_id, label)
        out += '}'
        return out
