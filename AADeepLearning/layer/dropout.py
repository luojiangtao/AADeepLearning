import numpy as np


class Dropout:
    """
    Dropout层
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
        if is_train:  # 只有训练时才用dropout
            assert (layer["drop_rate"] >= 0 and layer["drop_rate"] <= 1)
            mask = np.random.rand(flow_data.shape[0], flow_data.shape[1])
            # 生成 0 1 掩码
            layer['dropout'] = np.ones((flow_data.shape[0], flow_data.shape[1])) * (np.array(mask > layer["drop_rate"]))
            # rescale: 输出期望值 = （1 - drop_rate）*原始期望值 / （1 - drop_rate）  保持平均值不变
            flow_data = flow_data * layer['dropout'] / (1 - layer["drop_rate"])
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
        # rescale: 输出期望值 = （1 - drop_rate）*原始期望值 / （1 - drop_rate）  保持平均值不变
        flow_data = flow_data * layer['dropout'] / (1 - layer["drop_rate"])
        return flow_data, layer

    @staticmethod
    def update_parameters(layer, config, iteration):
        """
        更新权重和偏置项，这里无操作
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param config:配置
        :param iteration:迭代次数
        :return: 更新后的层
        """
        return layer
