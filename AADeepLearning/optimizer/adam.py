from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

class Adam:
    """
    Adam 优化器， 相当于 RMSprop + Momentum
    """
    @staticmethod
    def update_parameters(layer, keys, learning_rate, iteration):
        """
        更新权重和偏置项
        :param layer:  层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param keys: 要更新的键名
        :param learning_rate: 学习率
        :param iteration: 迭代次数
        :return: 更新后的层
        """
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8  # 防止除数等于零

        temp = {}
        for key in keys:
            temp["V_d"+key] = np.zeros(layer[key].shape)
            temp["S_d"+key] = np.zeros(layer[key].shape)

        for key in keys:
            temp["V_d"+key] = beta_1*temp["V_d"+key]+(1-beta_1)*layer["d"+key]
            temp["S_d"+key] = beta_2*temp["S_d"+key]+(1-beta_2)*layer["d"+key]**2
            V_corrected = temp["V_d"+key]/(1-np.power(beta_1, iteration))
            S_corrected = temp["S_d"+key]/(1-np.power(beta_2, iteration))
            layer[key] -= learning_rate*(V_corrected/np.sqrt(S_corrected+epsilon))

        return layer
