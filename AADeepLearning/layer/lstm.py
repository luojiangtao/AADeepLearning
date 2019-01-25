# 参考：https://www.cnblogs.com/pinard/p/6519110.html
import numpy as np
from ..optimizer.adam import Adam
from ..optimizer.momentum import Momentum
from ..optimizer.rmsprop import Rmsprop
from ..optimizer.sgd import Sgd


class LSTM:
    @staticmethod
    def init(layer, flow_data_shape, config):
        sequence_length = int(flow_data_shape["sequence_length"])
        neurons_number = layer['neurons_number']
        # 何凯明初始化，主要针对relu激活函数
        if layer["weight_init"] == 'msra':
            layer["Wf"] = np.random.randn(neurons_number, neurons_number) * (np.sqrt(2 / neurons_number))
            layer["Uf"] = np.random.randn(neurons_number, sequence_length) * (np.sqrt(2 / sequence_length))
            layer["Wi"] = np.random.randn(neurons_number, neurons_number) * (np.sqrt(2 / neurons_number))
            layer["Ui"] = np.random.randn(neurons_number, sequence_length) * (np.sqrt(2 / sequence_length))
            layer["Wa"] = np.random.randn(neurons_number, neurons_number) * (np.sqrt(2 / neurons_number))
            layer["Ua"] = np.random.randn(neurons_number, sequence_length) * (np.sqrt(2 / sequence_length))
            layer["Wo"] = np.random.randn(neurons_number, neurons_number) * (np.sqrt(2 / neurons_number))
            layer["Uo"] = np.random.randn(neurons_number, sequence_length) * (np.sqrt(2 / sequence_length))
            layer["V"] = np.random.randn(sequence_length, neurons_number) * (np.sqrt(2 / neurons_number))
        # xavier，主要针对tanh激活函数
        elif layer["weight_init"] == 'xavier':
            layer["Wf"] = np.random.randn(neurons_number, neurons_number) * (np.sqrt(1 / neurons_number))
            layer["Uf"] = np.random.randn(neurons_number, sequence_length) * (np.sqrt(1 / sequence_length))
            layer["Wi"] = np.random.randn(neurons_number, neurons_number) * (np.sqrt(1 / neurons_number))
            layer["Ui"] = np.random.randn(neurons_number, sequence_length) * (np.sqrt(1 / sequence_length))
            layer["Wa"] = np.random.randn(neurons_number, neurons_number) * (np.sqrt(1 / neurons_number))
            layer["Ua"] = np.random.randn(neurons_number, sequence_length) * (np.sqrt(1 / sequence_length))
            layer["Wo"] = np.random.randn(neurons_number, neurons_number) * (np.sqrt(1 / neurons_number))
            layer["Uo"] = np.random.randn(neurons_number, sequence_length) * (np.sqrt(1 / sequence_length))
            layer["V"] = np.random.randn(sequence_length, neurons_number) * (np.sqrt(1 / neurons_number))
        else:
            layer["Wf"] = np.random.randn(neurons_number, neurons_number) * 0.01
            layer["Uf"] = np.random.randn(neurons_number, sequence_length) * 0.01
            layer["Wi"] = np.random.randn(neurons_number, neurons_number) * 0.01
            layer["Ui"] = np.random.randn(neurons_number, sequence_length) * 0.01
            layer["Wa"] = np.random.randn(neurons_number, neurons_number) * 0.01
            layer["Ua"] = np.random.randn(neurons_number, sequence_length) * 0.01
            layer["Wo"] = np.random.randn(neurons_number, neurons_number) * 0.01
            layer["Uo"] = np.random.randn(neurons_number, sequence_length) * 0.01
            layer["V"] = np.random.randn(sequence_length, neurons_number) * 0.01
        layer["bf"] = np.zeros((neurons_number, 1))
        layer["bi"] = np.zeros((neurons_number, 1))
        layer["ba"] = np.zeros((neurons_number, 1))
        layer["bo"] = np.zeros((neurons_number, 1))
        layer["c"] = np.zeros((sequence_length, 1))
        flow_data_shape = {
            "flatten_size": sequence_length,
            "batch_size": flow_data_shape["batch_size"]
        }
        return layer, flow_data_shape

    @staticmethod
    def forword(flow_data, layer, is_train):
        ht = np.zeros((layer['neurons_number'], flow_data.shape[0]))
        layer["cache_ht_-1"] = ht
        ct = np.zeros((layer['neurons_number'], flow_data.shape[0]))
        layer["cache_ct_-1"] = ht
        for i in range(flow_data.shape[1]):
            xt = flow_data[:, i]
            layer["cache_xt_" + str(i)] = xt

            # 遗忘门 forget
            ft_1 = np.dot(layer["Wf"], ht)
            layer["cache_ft_1_" + str(i)] = ft_1
            ft_2 = np.dot(layer["Uf"], xt.T)
            layer["cache_ft_2_" + str(i)] = ft_2
            ft_3 = ft_1 + ft_2 + layer["bf"]
            layer["cache_ft_3_" + str(i)] = ft_3
            # ft = Sigmoid.forword(ft_3)
            ft = 1 / (1 + np.exp(-ft_3))
            layer["cache_ft_" + str(i)] = ft

            # 输入门1 input
            it_1 = np.dot(layer["Wi"], ht)
            layer["cache_it_1_" + str(i)] = it_1
            it_2 = np.dot(layer["Ui"], xt.T)
            layer["cache_it_2_" + str(i)] = it_2
            it_3 = it_1 + it_2 + layer["bi"]
            layer["cache_it_3_" + str(i)] = it_3
            # it = Sigmoid.forword(it_3)
            it = 1 / (1 + np.exp(-it_3))
            layer["cache_it_" + str(i)] = it

            # 输入门2
            at_1 = np.dot(layer["Wa"], ht)
            layer["cache_at_1_" + str(i)] = at_1
            at_2 = np.dot(layer["Ua"], xt.T)
            layer["cache_at_2_" + str(i)] = at_2
            at_3 = at_1 + at_2 + layer["ba"]
            layer["cache_at_3_" + str(i)] = at_3
            # at = Tanh.forword(at_3, layer, is_train)
            at = np.tanh(at_3)
            layer["cache_at_" + str(i)] = at

            # 细胞状态更新
            ct_1 = ct * ft
            layer["cache_ct_1_" + str(i)] = ct_1
            ct_2 = it * at
            layer["cache_ct_2_" + str(i)] = ct_2
            ct = ct_1 + ct_2
            layer["cache_ct_" + str(i)] = ct

            ot_1 = np.dot(layer["Wo"], ht)
            layer["cache_ot_1_" + str(i)] = ot_1
            ot_2 = np.dot(layer["Uo"], xt.T)
            layer["cache_ot_2_" + str(i)] = ot_2
            ot_3 = ot_1 + ot_2 + layer["bo"]
            layer["cache_ot_3_" + str(i)] = ot_3
            # ot = Sigmoid.forword(ot_3)
            ot = 1 / (1 + np.exp(-ot_3))
            layer["cache_ot_" + str(i)] = ot

            # 输出门
            # ht_1 = Tanh.forword(ct)
            ht_1 = np.tanh(ct)
            layer["cache_ht_1_" + str(i)] = ht_1
            ht = ot * ht_1
            layer["cache_ht_" + str(i)] = ht
        flow_data = np.dot(layer["V"], ht) + layer["c"]
        return flow_data, layer

    @staticmethod
    def backword(flow_data, layer, config):
        sequence_number = layer['input'].shape[1]
        ct = np.zeros(layer["cache_ct_0"].shape)
        layer["dc"] = np.sum(flow_data, axis=1, keepdims=True)
        layer["dV"] = np.dot(flow_data, layer["cache_ht_" + str(sequence_number - 1)].T)
        ht = np.dot(layer["V"].T, flow_data)
        output = np.zeros(layer["input"].shape)

        layer["dbo"] = np.zeros(layer["bo"].shape)
        layer["dWo"] = np.zeros(layer["Wo"].shape)
        layer["dUo"] = np.zeros(layer["Uo"].shape)

        layer["dba"] = np.zeros(layer["ba"].shape)
        layer["dWa"] = np.zeros(layer["Wa"].shape)
        layer["dUa"] = np.zeros(layer["Ua"].shape)

        layer["dbi"] = np.zeros(layer["bi"].shape)
        layer["dWi"] = np.zeros(layer["Wi"].shape)
        layer["dUi"] = np.zeros(layer["Ui"].shape)

        layer["dbf"] = np.zeros(layer["bf"].shape)
        layer["dWf"] = np.zeros(layer["Wf"].shape)
        layer["dUf"] = np.zeros(layer["Uf"].shape)
        for i in reversed(range(0, sequence_number)):
            ht_1 = ht * layer["cache_ot_" + str(i)]
            # ct = ct + Tanh.backword(ht_1, layer["cache_ht_1_" + str(i)])
            # dtanh/dz  = 1-a^2
            ct = ct + ht_1 * (1 - np.power(layer["cache_ht_1_" + str(i)], 2))
            ct_1 = ct
            ct = ct_1 * layer["cache_ft_" + str(i)]

            ot = ht * layer["cache_ht_1_" + str(i)]
            # ot_3 = Sigmoid.backword(ot, layer["cache_ot_" + str(i)])
            # dsigmoid/dz  = a*(1-a)
            ot_3 = ot * (layer["cache_ot_" + str(i)]*(1-layer["cache_ot_" + str(i)]))
            layer["dbo"] += np.sum(ot_3, axis=1, keepdims=True)
            layer["dWo"] += np.dot(ot_3, layer["cache_ht_" + str(i)].T)
            layer["dUo"] += np.dot(ot_3, layer["cache_xt_" + str(i)])
            ot_2 = ot_3
            ot_1 = ot_3

            ct_2 = ct
            at = ct_2 * layer["cache_it_" + str(i)]
            # at_3 = Tanh.backword(at, layer["cache_at_" + str(i)])
            # dtanh/dz  = 1-a^2
            at_3 = at * (1 - np.power(layer["cache_at_" + str(i)], 2))
            layer["dba"] += np.sum(at_3, axis=1, keepdims=True)
            layer["dWa"] += np.dot(at_3, layer["cache_ht_" + str(i)].T)
            layer["dUa"] += np.dot(at_3, layer["cache_xt_" + str(i)])
            at_1 = at_3
            at_2 = at_3

            it = ct_2 * layer["cache_at_" + str(i)]
            # it_3 = Sigmoid.backword(it, layer["cache_it_" + str(i)])
            # dsigmoid/dz  = a*(1-a)
            it_3 = ot * (layer["cache_it_" + str(i)]*(1-layer["cache_it_" + str(i)]))
            layer["dbi"] += np.sum(it_3, axis=1, keepdims=True)
            layer["dWi"] += np.dot(it_3, layer["cache_ht_" + str(i)].T)
            layer["dUi"] += np.dot(it_3, layer["cache_xt_" + str(i)])
            it_2 = it_3
            it_1 = it_3

            ft = ct_1 * layer["cache_ct_" + str(i)]
            # ft_3 = Sigmoid.backword(ft, layer["cache_ft_" + str(i)])
            # dsigmoid/dz  = a*(1-a)
            ft_3 = ft * (layer["cache_ft_" + str(i)]*(1-layer["cache_ft_" + str(i)]))
            layer["dbf"] += np.sum(ft_3, axis=1, keepdims=True)
            layer["dWf"] += np.dot(ft_3, layer["cache_ht_" + str(i)].T)
            layer["dUf"] += np.dot(ft_3, layer["cache_xt_" + str(i)])
            ft_2 = ft_3
            ft_1 = ft_3

            xt = np.dot(layer["Uf"].T, ft_2) + np.dot(layer["Ui"].T, it_2) + np.dot(layer["Ua"].T, at_2) + np.dot(
                layer["Uo"].T, ot_2)
            ht = np.dot(layer["Wf"].T, ft_1) + np.dot(layer["Wi"].T, it_1) + np.dot(layer["Wa"].T, at_1) + np.dot(
                layer["Wo"].T, ot_1)

            output[:, i] = xt.T
        return output, layer

    # @staticmethod
    # def update_parameters(layer, config, iteration):
    #     layer["Wf"] -= config["learning_rate"] * layer["dWf"]
    #     layer["Uf"] -= config["learning_rate"] * layer["dUf"]
    #     layer["Wi"] -= config["learning_rate"] * layer["dWi"]
    #     layer["Ui"] -= config["learning_rate"] * layer["dUi"]
    #     layer["Wa"] -= config["learning_rate"] * layer["dWa"]
    #     layer["Ua"] -= config["learning_rate"] * layer["dUa"]
    #     layer["Wo"] -= config["learning_rate"] * layer["dWo"]
    #     layer["Uo"] -= config["learning_rate"] * layer["dUo"]
    #     layer["V"] -= config["learning_rate"] * layer["dV"]
    #     layer["bf"] -= config["learning_rate"] * layer["dbf"]
    #     layer["bi"] -= config["learning_rate"] * layer["dbi"]
    #     layer["ba"] -= config["learning_rate"] * layer["dba"]
    #     layer["bo"] -= config["learning_rate"] * layer["dbo"]
    #     layer["c"] -= config["learning_rate"] * layer["dc"]
    #     return layer

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
        keys = ['Wf', 'Uf', 'Wi', 'Ui', 'Wa', 'Ua', 'Wo', 'Uo', 'V', 'bf', 'bi', 'ba', 'bo', 'c']
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