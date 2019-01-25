"""Contains the base Layer class, from which all layers inherit.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np


class Rmsprop:
    """
    学习率自适应优化器
    """
    @staticmethod
    def update_parameters(layer, keys, learning_rate, decay=0.9):
        """
        更新参数
        :param layer:  层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param keys: 需要更新的键名
        :param learning_rate: 学习率
        :param decay: 衰减系数
        :return:  更新后的层
        """
        # 防止除数为0
        epsilon = 1e-8
        temp = {}
        for key in keys:
            temp["S_d" + key] = np.zeros(layer[key].shape)

        for key in keys:
            temp["S_d" + key] = decay * temp["S_d" + key] + (1 - decay) * layer["d" + key] ** 2
            layer[key] -= (learning_rate / (np.sqrt(temp["S_d" + key] + epsilon))) * layer["d" + key]
        return layer
