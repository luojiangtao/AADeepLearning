import numpy as np
from ..optimizer.adam import Adam
from ..optimizer.momentum import Momentum
from ..optimizer.rmsprop import Rmsprop
from ..optimizer.sgd import Sgd


class FullyConnected:
    """
    全连接层
    """
    @staticmethod
    def init(layer, flow_data_shape, config):
        """
        初始化
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param flow_data_shape: 流动数据的形状
        :param config:配置
        :return: 更新后的层， 流动数据的形状
        """
        flatten_size = int(flow_data_shape["flatten_size"])
        if layer["weight_init"] == 'msra':
            # 何凯明初始化，主要针对relu激活函数
            layer["W"] = np.random.randn(layer['neurons_number'],
                                              flatten_size) * (np.sqrt(2 / flatten_size))
        elif layer["weight_init"] == 'xavier':
            # xavier，主要针对tanh激活函数
            layer["W"] = np.random.randn(layer['neurons_number'],
                                              flatten_size) * (np.sqrt(1 / flatten_size))
        else:
            # 高斯初始化
            layer["W"] = np.random.randn(layer['neurons_number'], flatten_size) * 0.01
        layer["b"] = np.zeros((layer['neurons_number'], 1))

        flow_data_shape = {
            "flatten_size": layer['neurons_number'],
            "batch_size": flow_data_shape["batch_size"]
        }

        print(layer['name']+",W.shape：", layer["W"].shape)
        print(layer['name']+",b.shape：", layer["b"].shape)

        return layer, flow_data_shape

    @staticmethod
    def forword(flow_data, layer, is_train):
        """
        前向传播
        :param flow_data: 流动数据
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param is_train: 是否是训练模式
        :return: 流动数据， 更新后的层
        """
        flow_data = np.dot(layer['W'], flow_data) + layer['b']
        return flow_data, layer
    @staticmethod
    def backword(flow_data, layer, config):
        """
        反向传播
        :param flow_data: 流动数据
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param config:配置
        :return: 流动数据， 更新后的层
        """
        layer["dW"] = (1 / config['batch_size']) * np.dot(flow_data, layer['input'].T)
        layer["db"] = (1 / config['batch_size']) * np.sum(flow_data, axis=1, keepdims=True)
        # dx
        flow_data = np.dot(layer['W'].T, flow_data)
        return flow_data, layer

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
        keys = ['W', 'b']
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
