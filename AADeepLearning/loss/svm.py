import numpy as np

class SVM:
    """
    SVM损失层，又称为Hinge损失函数，一般用于最后一层分类
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
        前向传播，这里没有操作，直接计算损失
        :param flow_data: 流动数据
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param is_train: 是否是训练模式
        :return: 流动数据， 更新后的层
        """
        # for i in range(config['batch_size']):
        # for i in range(flow_data.shape[1]):
        #     flow_data[:, i] = np.exp(flow_data[:, i]) / np.sum(np.exp(flow_data[:, i]))
        return flow_data, layer
    @staticmethod
    def compute_cost(flow_data, layer, label):
        """
        计算代价（SVM损失，又称为Hinge损失）
        :param flow_data: 前向传播最后一层输出
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param label: 标签
        :return: 损失
        """
        delta = 0.2
        if 'delta' in layer.keys():
            delta = layer['delta']
        flow_data = flow_data.T
        batch_size = label.shape[0]
        loss = 0.0
        for i in range(batch_size):
            # loss = max(0, 错误得分 - 正确得分 + delta)
            # 正确类别索引
            right_index = np.argmax(label[i])
            # # 正确类别值
            positive_x = flow_data[i][right_index]
            # 代入hinge loss公式
            temp = flow_data[i] - positive_x + delta
            # 剔除正确类里面的值
            temp[right_index] = 0
            # 小于零就转换为0， 大于零不变 相当于：temp=max(0, temp)
            temp = temp * np.array(temp > 0)
            loss += np.sum(temp)

        loss = loss / batch_size
        return loss
    @staticmethod
    def backword(flow_data, layer, label):
        """
        反向传播
        :param flow_data: 流动数据
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param label: 标签
        :return: 流动数据， 更新后的层
        """
        delta = 0.2
        if 'delta' in layer.keys():
            delta = layer['delta']
        flow_data = flow_data.T
        batch_size = label.shape[0]
        output = np.zeros(flow_data.shape)
        for i in range(batch_size):
            # loss += -np.sum(np.dot(batch_label[i], np.log(flow_data[:, i])))
            # loss = max(0, 错误得分 - 正确得分 + delta)
            # 正确类别索引
            right_index = np.argmax(label[i])
            # # 正确类别值
            positive_x = flow_data[i][right_index]
            # 代入hinge loss公式
            temp = flow_data[i] - positive_x + delta
            # 剔除正确类里面的值
            temp[right_index] = 0
            # 小于零就转换为0， 大于零转行为1, 0 1掩码
            temp = np.ones(temp.shape) * np.array(temp > 0)
            # 正确位置的梯度
            temp[right_index] = -np.sum(temp)
            output[i] = temp

        # 获取最末层误差信号,反向传播
        # print(output[0])
        # print(output.shape)
        # exit()
        return output.T

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
