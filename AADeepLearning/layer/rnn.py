import numpy as np

from ..optimizer.adam import Adam
from ..optimizer.momentum import Momentum
from ..optimizer.rmsprop import Rmsprop
from ..optimizer.sgd import Sgd

class RNN:
    @staticmethod
    def init(layer, flow_data_shape, config):
        sequence_length = int(flow_data_shape["sequence_length"])
        # 何凯明初始化，主要针对relu激活函数
        if layer["weight_init"] == 'msra':
            layer["U"] = np.random.randn(layer['neurons_number'], flow_data_shape["sequence_length"]) * (
                np.sqrt(2 / sequence_length))
            layer["W"] = np.random.randn(layer['neurons_number'], layer['neurons_number']) * (
                np.sqrt(2 / layer['neurons_number']))
            layer["V"] = np.random.randn(flow_data_shape["sequence_length"], layer['neurons_number']) * (
                np.sqrt(2 / layer['neurons_number']))
        # xavier，主要针对tanh激活函数
        elif layer["weight_init"] == 'xavier':
            layer["U"] = np.random.randn(layer['neurons_number'], flow_data_shape["sequence_length"]) * (
                np.sqrt(1 / sequence_length))
            layer["W"] = np.random.randn(layer['neurons_number'], layer['neurons_number']) * (
                np.sqrt(1 / layer['neurons_number']))
            layer["V"] = np.random.randn(flow_data_shape["sequence_length"], layer['neurons_number']) * (
                np.sqrt(1 / layer['neurons_number']))
        else:
            layer["U"] = np.random.randn(layer['neurons_number'], flow_data_shape["sequence_length"]) * 0.01
            layer["W"] = np.random.randn(layer['neurons_number'], layer['neurons_number']) * 0.01
            layer["V"] = np.random.randn(flow_data_shape["sequence_length"], layer['neurons_number']) * 0.01
        layer["bW"] = np.zeros((layer['neurons_number'], 1))
        layer["bV"] = np.zeros((flow_data_shape["sequence_length"], 1))
        flow_data_shape = {
            "flatten_size": flow_data_shape["sequence_length"],
            "batch_size": flow_data_shape["batch_size"]
        }
        return layer, flow_data_shape

    @staticmethod
    def forword(layer, flow_data, is_train):
        # flow_data = flow_data[0]
        h = np.zeros((layer['neurons_number'], flow_data.shape[0]))
        for i in range(flow_data.shape[1]):
            sequence = flow_data[:, i]
            layer["U_input_" + str(i)] = sequence
            U_multiply_X = np.dot(layer["U"], sequence.T)
            layer["W_input_" + str(i)] = h
            W_multiply_h = np.dot(layer["W"], h)
            h = U_multiply_X + W_multiply_h
            h = h + layer["bW"]
            h = np.tanh(h)
            layer["tanh_output"] = h
        # 缓存该层的输入
        layer["V_input"] = h
        flow_data = np.dot(layer["V"], h) + layer["bV"]
        return flow_data, layer

    @staticmethod
    def backword(flow_data, layer, config):
        output_all = np.zeros(layer["input"].shape)
        layer["dW"] = np.zeros(layer["W"].shape)
        layer["dU"] = np.zeros(layer["U"].shape)
        layer["dbW"] = np.zeros(layer["bW"].shape)
        layer["dbV"] = np.sum(flow_data, axis=1, keepdims=True)
        layer["dV"] = np.dot(flow_data, layer['V_input'].T)
        h = np.dot(layer["V"].T, flow_data)
        for i in reversed(range(0, layer['input'].shape[1])):
            # tanh 梯度
            h = h * (1 - np.power(layer["tanh_output"], 2))
            layer["dbW"] += np.sum(h, axis=1, keepdims=True)
            layer["dW"] += np.dot(h, layer["W_input_" + str(i)].T)
            layer["dU"] += np.dot(h, layer["U_input_" + str(i)])

            output_all[:, i] = np.dot(h.T, layer["U"])
            h = np.dot(layer["W"].T, h)
        return output_all, layer

    @staticmethod
    def update_parameters(layer, config, iteration):
        """
        更新权重和偏置项
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param config:配置
        :param iteration:迭代次数
        :return: 更新后的层
        """
        # 要更新的键名
        keys = ['U', 'W', 'V', 'bW', 'bV']
        if "optimizer" in config.keys() and config["optimizer"] == 'momentum':
            layer = Momentum.update_parameters(layer, keys, config['learning_rate'], config['momentum_coefficient'])
        elif "optimizer" in config.keys() and config["optimizer"] == 'rmsprop':
            layer = Rmsprop.update_parameters(layer, keys, config['learning_rate'])
        elif "optimizer" in config.keys() and config["optimizer"] == 'adam':
            layer = Adam.update_parameters(layer, keys, config['learning_rate'], iteration)
        else:
            # 默认使用 sgd
            layer = Sgd.update_parameters(layer, keys, config['learning_rate'])
        return layer