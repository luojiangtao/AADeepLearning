from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np


class Momentum:
    """
    动量优化器
    """
    @staticmethod
    def update_parameters(layer, keys, learning_rate, momentum_coefficient):
        """
        更新参数
        :param layer:  层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param keys: 需要更新的键名
        :param learning_rate: 学习率
        :param momentum_coefficient: 动量系数
        :return:  更新后的层
        """
        temp = {}
        for key in keys:
            temp["V_d"+key] = np.zeros(layer[key].shape)

        for key in keys:
            temp["V_d"+key] = momentum_coefficient*temp["V_d"+key]+layer["d"+key]
            layer[key] -= learning_rate*temp["V_d"+key]
        return layer
