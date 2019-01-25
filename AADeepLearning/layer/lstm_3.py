# https://blog.csdn.net/wjc1182511338/article/details/79285503
import numpy as np
from .activation.tanh import Tanh
from .activation.sigmoid import Sigmoid


class LSTM:
    @staticmethod
    def init(layer, flow_data_shape):
        sequence_length = int(flow_data_shape["sequence_length"])
        # forget 遗忘门
        layer["weight_f"] = np.random.randn(layer['neurons_number'], flow_data_shape["sequence_length"])
        # input
        layer["weight_i"] = np.random.randn(layer['neurons_number'], flow_data_shape["sequence_length"])
        # current inputstate
        layer["weight_c"] = np.random.randn(layer['neurons_number'], flow_data_shape["sequence_length"])
        # output
        layer["weight_o"] = np.random.randn(layer['neurons_number'], flow_data_shape["sequence_length"])
        layer["bias_f"] = np.zeros((layer['neurons_number'], 1))
        layer["bias_i"] = np.zeros((layer['neurons_number'], 1))
        layer["bias_c"] = np.zeros((layer['neurons_number'], 1))
        layer["bias_o"] = np.zeros((layer['neurons_number'], 1))
        flow_data_shape = {
            "flatten_size": flow_data_shape["sequence_length"],
            "batch_size": flow_data_shape["batch_size"]
        }
        return layer, flow_data_shape

    @staticmethod
    def forword(layer, flow_data):
        # flow_data = flow_data[0]
        ht = np.zeros((layer['neurons_number'], flow_data.shape[0]))
        ct = np.zeros((layer['neurons_number'], flow_data.shape[0]))
        for i in range(flow_data.shape[1]):
            xt = flow_data[:, i]
            ft = Sigmoid.forword(np.dot(layer["weight_f"], np.concatenate(ht, xt)) + layer['bias_f'])
            it = Sigmoid.forword(np.dot(layer["weight_i"], np.concatenate(ht, xt)) + layer['bias_i'])
            _ct = Tanh.forword(np.dot(layer["weight_c"], np.concatenate(ht, xt)) + layer['bias_c'])

            ct = ft * ct + it * _ct
            ot = Sigmoid.forword(np.dot(layer["weight_o"], np.concatenate(ht, xt)) + layer['bias_o'])
            ht = ot * Tanh.forword(ct)
        # 缓存该层的输入
        # todo 可能还有 weight_V
        # layer["weight_V_input"] = h
        # flow_data = np.dot( layer["weight_V"],h) + layer["bias_V"]
        # print(flow_data.shape)
        # exit()
        # print(flow_data.shape)
        # exit()
        return flow_data, layer

    @staticmethod
    def backword(flow_data, layer, config):
        output_all = np.zeros(layer["input"].shape)
        # print(output_all.shape)
        # exit()
        layer["weight_W_gradient"] = np.zeros(layer["weight_W"].shape)
        layer["weight_U_gradient"] = np.zeros(layer["weight_U"].shape)
        layer["bias_W_gradient"] = np.zeros(layer["bias_W"].shape)
        # todo 可能要列相加
        layer["bias_V_gradient"] = flow_data
        layer["weight_V_gradient"] = np.dot(flow_data, layer['weight_V_input'].T)
        h = np.dot(layer["weight_V"].T, flow_data)
        for i in reversed(range(0, layer['input'].shape[1])):
            h = Tanh.backword(h, layer)
            layer["bias_W_gradient"] += np.sum(h, axis=1, keepdims=True)
            # print(h.shape)
            # print(layer["weight_W_input_"+str(i)].T.shape)
            # print(layer["weight_W_gradient"].shape)
            # print("----------")
            # exit()
            layer["weight_W_gradient"] += np.dot(h, layer["weight_W_input_" + str(i)].T)
            layer["weight_U_gradient"] += np.dot(h, layer["weight_U_input_" + str(i)])

            output_all[:, i] = np.dot(h.T, layer["weight_U"])
            h = np.dot(layer["weight_W"].T, h)

        # print(output_all.shape)
        # exit()
        return layer, output_all
