import numpy as np


class BatchNormalization:
    """
    批归一化
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
        layer["gamma"] = np.ones((flatten_size, 1))
        layer["beta"] = np.zeros((flatten_size, 1))

        print(layer['name'] + ",gamma.shape：", layer["gamma"].shape)
        print(layer['name'] + ",beta.shape：", layer["beta"].shape)

        return layer, flow_data_shape

    @staticmethod
    def forword(flow_data, layer, is_train):
        """
        前向传播
        :param flow_data: 流动数据
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :return: 流动数据， 更新后的层
        """
        epsilon = 1e-8
        layer['mean'] = np.mean(flow_data, axis=1, keepdims=True)
        layer['std'] = np.std(flow_data, axis=1, keepdims=True)
        layer['norm'] = (flow_data - layer['mean']) / (layer['std'] + epsilon)
        flow_data = layer["gamma"] * layer['norm'] + layer["beta"]
        return flow_data, layer

    @staticmethod
    def backword(flow_data, layer, config):
        """
        反向传播
        :param flow_data: 流动数据
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param config: 配置
        :return: 流动数据， 更新后的层
        """
        epsilon = 1e-8
        # gamma 的梯度
        layer["dgamma"] = np.sum(flow_data * layer['norm'], axis=1, keepdims=True)
        # beta 的梯度
        layer["dbeta"] = np.sum(flow_data, axis=1, keepdims=True)
        flow_data = (layer["gamma"] / (layer['std'] + epsilon)) * (
                flow_data - layer["dgamma"] * layer['norm'] / config['batch_size'] - np.mean(flow_data, axis=1,
                                                                                             keepdims=True))
        return flow_data, layer

    @staticmethod
    def update_parameters(layer, config, iteration):
        """
        更新参数
        :param layer:  层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param config: 配置
        :param iteration: 迭代次数
        :return:
        """
        layer["gamma"] -= config["learning_rate"] * layer["dgamma"]
        layer["beta"] -= config["learning_rate"] * layer["dbeta"]
        return layer
