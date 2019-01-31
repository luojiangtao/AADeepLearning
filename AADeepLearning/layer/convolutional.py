import numpy as np
from ..optimizer.adam import Adam
from ..optimizer.momentum import Momentum
from ..optimizer.rmsprop import Rmsprop
from ..optimizer.sgd import Sgd


class Convolutional:
    """
    卷积层
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
        # 何凯明初始化，主要针对relu激活函数
        if layer["weight_init"] == 'msra':
            layer["W"] = np.random.randn(layer['kernel_number'], flow_data_shape['channel'], layer['kernel_height'],
                                         layer['kernel_width']) * (
                             np.sqrt(2 / (flow_data_shape['channel'] * layer['kernel_height'] * layer['kernel_width'])))
        # xavier，主要针对tanh激活函数
        elif layer["weight_init"] == 'xavier':
            layer["W"] = np.random.randn(layer['kernel_number'], flow_data_shape['channel'], layer['kernel_height'],
                                         layer['kernel_width']) * (
                             np.sqrt(1 / (flow_data_shape['channel'] * layer['kernel_height'] * layer['kernel_width'])))
        else:
            layer["W"] = np.random.randn(layer['kernel_number'], flow_data_shape['channel'], layer['kernel_height'],
                                         layer['kernel_width']) * 0.01
        layer["b"] = np.zeros((layer['kernel_number'], 1, 1, 1))
        flow_data_shape = {
            "batch_size": flow_data_shape['batch_size'],
            "channel": layer['kernel_number'],
            "height": ((flow_data_shape['height'] + layer['padding'] * 2 - layer['kernel_height'])) // layer[
                'stride'] + 1,
            "width": ((flow_data_shape['width'] + layer['padding'] * 2 - layer['kernel_width']) // layer['stride']) + 1
        }

        print(layer['name'] + ",W.shape：", layer["W"].shape)
        print(layer['name'] + ",b.shape：", layer["b"].shape)

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
        padding = layer['padding']
        if padding != 0:
            flow_data = Convolutional.padding(flow_data, padding)
            layer['padding_input'] = flow_data
        kernel_height = layer['kernel_height']
        kernel_width = layer['kernel_width']

        batch_size = flow_data.shape[0]
        output_height = ((flow_data.shape[2] - kernel_width) // layer['stride']) + 1
        output_width = ((flow_data.shape[3] - kernel_height) // layer['stride']) + 1

        # 卷积输出
        output = np.zeros((batch_size, layer['kernel_number'], output_height, output_width))
        # 开始卷积
        for channel in range(output.shape[1]):  # 遍历输出的通道数，输出的通道数等于卷积核的个数
            for height in range(output.shape[2]):  # 遍历输出的高
                for width in range(output.shape[3]):  # 遍历输出的宽
                    # 滑动窗口截取部分
                    sliding_window = flow_data[:,:,
                                     height * layer['stride']:height * layer['stride'] + kernel_height,
                                     width * layer['stride']:width * layer['stride'] + kernel_width
                                     ]
                    output[:,channel,height,width] = np.sum(np.sum(np.sum((sliding_window * layer["W"][channel]) + layer["b"][channel], axis=2), axis=2), axis=1)
        return output, layer

    @staticmethod
    def backword(flow_data, layer, config):
        """
        反向传播
        :param flow_data: 流动数据
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param config:配置
        :return: 流动数据， 更新后的层
        """
        layer["dW"] = np.zeros(layer['W'].shape)
        layer["db"] = np.zeros(layer['b'].shape)
        kernel_height = layer['kernel_height']
        kernel_width = layer['kernel_width']

        if layer['padding'] != 0:
            forword_input = layer['padding_input']
        else:
            forword_input = layer['input']

        output = np.zeros((forword_input.shape))

        for channel in range(flow_data.shape[1]):  # 遍历输入梯度的通道数，输入梯度的通道数等于卷积核的个数
            for height in range(flow_data.shape[2]):  # 遍历输入梯度的高
                for width in range(flow_data.shape[3]):  # 遍历输入梯度的宽
                    # 前向传播输入数据，滑动截取窗口
                    sliding_window = forword_input[:,:,
                                     height * layer['stride']:height * layer['stride'] + kernel_height,
                                     width * layer['stride']:width * layer['stride'] + kernel_width
                                     ]
                    # dx
                    output[:,:,
                    height * layer['stride']:height * layer['stride'] + kernel_height,
                    width * layer['stride']:width * layer['stride'] + kernel_width
                    ] += flow_data[:,channel,height,width].reshape(flow_data.shape[0], 1, 1, 1) * layer['W'][channel]
                    # 单个卷积核梯度 = 前向输入数据的滑动窗口 * 梯度对应通道（卷积核），对应高宽
                    layer["dW"][channel] += np.mean(flow_data[:,channel,height,width].reshape(flow_data.shape[0], 1, 1, 1) * sliding_window, axis=0)

                    layer["db"][channel][0][0][0] += np.mean(flow_data[:,channel,height,width])
        if layer['padding'] != 0:
            output = Convolutional.delete_padding(output, layer['padding'])
        return output, layer

    @staticmethod
    def update_parameters(layer, config, iteration):
        """
        更新权重和偏置项
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param config:配置
        :param iteration:迭代次数
        :return: 更新后的层
        """
        # 需要更新的键名
        keys = ['W', 'b']
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

    @staticmethod
    def padding(flow_data, padding):
        """
        填充
        :param flow_data: 流动数据
        :param padding: 填充多少层0
        :return:
        """
        padding_flow_data = np.zeros((flow_data.shape[0], flow_data.shape[1], flow_data.shape[2] + padding * 2,
                                      flow_data.shape[3] + padding * 2))
        for batch in range(flow_data.shape[0]):  # 遍历总样本数
            for channel in range(flow_data.shape[1]):  # 遍历 通道数
                # 在二位矩阵外面填充 padding圈零
                padding_flow_data[batch][channel] = np.pad(flow_data[batch][channel],
                                                           ((padding, padding), (padding, padding)), 'constant')
        return padding_flow_data

    @staticmethod
    def delete_padding(flow_data, padding):
        """
        删除填充
        :param flow_data: 流动数据
        :param padding: 去掉外面多少层
        :return:
        """
        # 定义结构
        delete_padding_flow_data = np.zeros((flow_data.shape[0], flow_data.shape[1], flow_data.shape[2] - padding * 2,
                                             flow_data.shape[3] - padding * 2))
        for batch in range(flow_data.shape[0]):
            for channel in range(flow_data.shape[1]):
                height = flow_data[batch][channel].shape[0]
                width = flow_data[batch][channel].shape[1]
                # 对应位置复制过来
                delete_padding_flow_data[batch][channel] = flow_data[batch][channel][padding:height - padding,
                                                           padding:width - padding]
        return delete_padding_flow_data
