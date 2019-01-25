# https://blog.csdn.net/flyinglittlepig/article/details/72229041
import numpy as np
from .activation.tanh import Tanh
from .activation.sigmoid import Sigmoid


class LSTM():
    @staticmethod
    def init(layer, flow_data_shape):
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
    def forword(layer, flow_data):
        ht = np.zeros((layer['neurons_number'], flow_data.shape[0]))
        layer["cache_ht_0"] = ht
        ct = np.zeros((layer['neurons_number'], flow_data.shape[0]))
        layer["cache_ct_-1"] = ht
        for i in range(flow_data.shape[1]):
            xt = flow_data[:, i]
            layer["cache_xt_" + str(i)] = xt

            print(ht.shape)
            print(xt.shape)
            exit()
            ht_xt = np.concatenate(ht, xt)
            layer["cache_ht_xt_" + str(i)] = ht_xt

            ft_1 = np.dot(layer["Wf"], ht_xt) + layer["bf"]
            layer["cache_ft_1_" + str(i)] = ft_1
            ft = Sigmoid.forword(ft_1)
            layer["cache_ft_" + str(i)] = ft

            it_1 = np.dot(layer["Wi"], ht_xt) + layer["bi"]
            layer["cache_it_1_" + str(i)] = it_1
            it = Sigmoid.forword(it_1)
            layer["cache_it_" + str(i)] = it

            at_1 = np.dot(layer["Wa"], ht_xt) + layer["ba"]
            layer["cache_at_1_" + str(i)] = at_1
            at = Tanh.forword(at_1)
            layer["cache_at_" + str(i)] = at

            ot_1 = np.dot(layer["Wo"], ht_xt) + layer["bo"]
            layer["cache_ot_1_" + str(i)] = ot_1
            ot = Sigmoid.forword(ot_1)
            layer["cache_ot_" + str(i)] = ot

            ct_1 = ct * ft
            layer["cache_ct_1_" + str(i)] = ct_1
            ct_2 = it * at
            layer["cache_ct_2_" + str(i)] = ct_2
            ct = ct_1 + ct_2
            layer["cache_ct_" + str(i)] = ct

            ht_1 = Tanh.forword(ct)
            layer["cache_ht_1_" + str(i)] = ht_1
            ht = ot * ht_1
            layer["cache_ht_" + str(i)] = ht

        yt = np.dot(layer["Wy"], ht) + layer["by"]
        layer["cache_yt"] = yt
        flow_data = yt
        # print(flow_data.shape)
        # exit()
        # print(flow_data.shape)
        # exit()
        return flow_data, layer

    @staticmethod
    def backword(flow_data, layer, config):
        sequence_number = layer['input'].shape[1]
        layer["dy"] = flow_data
        layer["dWy"] = np.dot(flow_data, layer["cache_ht_" + str(sequence_number - 1)].T)
        ht = np.dot(layer["Wy"].T, flow_data)
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
        ct = np.zeros(layer["cache_ct_0"].shape)
        for i in reversed(range(0, sequence_number)):
            ht_1 = ht * layer["cache_ot_" + str(i)]
            ct = Tanh.backword(ht_1, layer["cache_ht_1_" + str(i)]) + ct
            ct_1 = ct
            ct = ct_1 * layer["cache_ft_" + str(i)]
            ct_2 = ct

            ot = ht * layer["cache_ht_1_" + str(i)]
            ot_1 = Sigmoid.backword(ot, layer["cache_ot_" + str(i)])
            layer["dbo"] += np.sum(ot_1, axis=1, keepdims=True)
            layer["dWo"] += np.dot(ot_1, layer["cache_ht_xt_" + str(i)].T)

            at = ct_2 * layer["cache_it_" + str(i)]
            at_1 = Tanh.backword(at, layer["cache_at_" + str(i)])
            layer["dba"] += np.sum(at_1, axis=1, keepdims=True)
            layer["dWa"] += np.dot(at_1, layer["cache_ht_xt_" + str(i)].T)

            it = ct_2 * layer["cache_at_" + str(i)]
            it_1 = Sigmoid.backword(it, layer["cache_it_" + str(i)])
            layer["dbi"] += np.sum(it_1, axis=1, keepdims=True)
            layer["dWi"] += np.dot(it_1, layer["cache_ht_xt_" + str(i)].T)

            ft = ct_1 * layer["cache_ct_" + str(i - 1)]
            ft_1 = Sigmoid.backword(ft, layer["cache_ft_" + str(i)])
            layer["dbf"] += np.sum(ft_1, axis=1, keepdims=True)
            layer["dWf"] += np.dot(ft_1, layer["cache_ht_xt_" + str(i)].T)

            # ht_xt = np.dot(layer["Uf"].T, ft_2) + np.dot(layer["Ui"].T, it_2) + np.dot(layer["Ua"].T, at_2) + np.dot(
            #     layer["Uo"].T, ot_2)
            ht_xt = np.dot(layer["Wf"].T, ft_1) + np.dot(layer["Wi"].T, it_1) + np.dot(layer["Wa"].T, at_1) + np.dot(
                layer["Wo"].T, ot_1)
            ht = ht_xt[:ht.shape[0]]
            xt = ht_xt[ht.shape[0]:]
            output[:, i] = xt.T
        return layer, output

    # 输出单元激活函数
    @staticmethod
    def softmax(x):
        x = np.array(x)
        max_x = np.max(x)
        return np.exp(x - max_x) / np.sum(np.exp(x - max_x))
