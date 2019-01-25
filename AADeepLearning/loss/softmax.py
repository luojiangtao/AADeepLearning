import numpy as np

class SoftMax:
    """
    SoftMax层 一般用于最后一层分类
    """
    @staticmethod
    def init(layer, flow_data_shape, config):
        """
        初始化， 这里无操作
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
        # print(flow_data.shape)
        # todo 不用循环
        # for i in range(config['batch_size']):
        for i in range(flow_data.shape[1]):
            flow_data[:, i] = np.exp(flow_data[:, i]) / np.sum(np.exp(flow_data[:, i]))
        return flow_data, layer
    @staticmethod
    def compute_cost(flow_data, layer, label):
        """
        计算代价（交叉熵损失）
        :param flow_data: 前向传播最后一层输出
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param label: 标签
        :return: 损失
        """
        batch_size = flow_data.shape[1]
        loss = 0.0
        for i in range(batch_size):
            loss += -np.sum(np.dot(label[i], np.log(flow_data[:, i])))
        loss = loss / batch_size
        return loss
    @staticmethod
    def backword(flow_data, layer, label):
        """
        反向传播
        :param flow_data: 流动数据
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param config:配置
        :return: 流动数据， 更新后的层
        """
        # 获取最末层误差信号 softmax反向传播
        return flow_data - label.T

    @staticmethod
    def update_parameters(layer, config, iteration):
        """
        更新权重和偏置项， 这里无操作
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param config:配置
        :param iteration:迭代次数
        :return: 更新后的层
        """
        return layer
