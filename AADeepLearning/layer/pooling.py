import numpy as np

class Pooling:
    """
    池化层
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
        flow_data_shape = {
            "batch_size": flow_data_shape['batch_size'],
            "channel": flow_data_shape['channel'],
            "height": (flow_data_shape['height'] - layer['kernel_height']) // layer['stride'] + 1,
            "width": (flow_data_shape['width'] - layer['kernel_width']) // layer['stride'] + 1
        }
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
        kernel_height = layer['kernel_height']
        kernel_width = layer['kernel_width']
        batch_size = flow_data.shape[0]
        channels = flow_data.shape[1]
        output_width = ((flow_data.shape[2] - kernel_width) // layer['stride']) + 1
        output_height = ((flow_data.shape[3] - kernel_height) // layer['stride']) + 1
        # 池化总输出
        pooling_out = np.zeros((batch_size, channels, output_width, output_height))
        # 开始池化
        for channel in range(channels):  # 遍历输出通道数
            for height in range(output_height):  # 遍历输出高
                for width in range(output_width):  # 遍历输出宽
                    # 滑动窗口截取部分
                    sliding_window = flow_data[:,channel,
                                     height * layer['stride']:height * layer['stride'] + kernel_height,
                                     width * layer['stride']:width * layer['stride'] + kernel_width
                                     ]
                    if 'mode' in layer.keys() and layer['mode'] == 'average':
                        # 平均池化
                        pooling_out[:,channel,height,width] = np.average(np.average(sliding_window,axis=1),axis=1)
                    else:
                        # 默认取最大值
                        pooling_out[:,channel,height,width] = np.max(np.max(sliding_window,axis=1),axis=1)

        return pooling_out, layer

    @staticmethod
    def backword(flow_data, layer, config):
        """
        反向传播
        :param flow_data: 流动数据
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param config:配置
        :return: 流动数据， 更新后的层
        """
        kernel_height = layer['kernel_height']
        kernel_width = layer['kernel_width']
        kernel_total = kernel_height*kernel_width
        stride = layer['stride']
        output = np.zeros(layer['input'].shape)
        for channel in range(flow_data.shape[1]):
            for height in range(flow_data.shape[2]):
                for width in range(flow_data.shape[3]):
                    if 'mode' in layer.keys() and layer['mode'] == 'average':
                        # 平均池化
                        output[:,channel,
                                         height * stride:height * stride + kernel_height,
                                         width * stride:width * stride + kernel_width
                                         ] += flow_data[:,channel,height,width].reshape(-1,1,1)/kernel_total
                    else:
                        # 滑动窗口截取部分
                        sliding_window = layer['input'][:,channel,
                                         height * stride:height * stride + kernel_height,
                                         width * stride:width * stride + kernel_width
                                         ]
                        # 求最大位置0 1 掩码
                        mask = np.ones(sliding_window.shape) * (
                                    np.max(np.max(sliding_window, axis=1), axis=1).reshape(-1, 1, 1) == sliding_window)
                        output[:,channel,
                        height * stride:height * stride + kernel_height,
                        width * stride:width * stride + kernel_width
                        ] += mask*(flow_data[:,channel,height,width].reshape(-1,1,1))
        return output, layer

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