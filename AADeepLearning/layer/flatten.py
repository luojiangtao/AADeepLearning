class Flatten:
    """
    展平数据，一般用于卷积层和全连接层中间
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
            "flatten_size": flow_data_shape["channel"] * flow_data_shape["height"] * flow_data_shape["width"],
            "batch_size": flow_data_shape["batch_size"]
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
        # X_train shape: (60000, 1, 28, 28) ——> (784, 60000)
        # 流动数据，一层一层的计算，并先后流动
        flow_data = flow_data.reshape(flow_data.shape[0], -1).T
        return flow_data, layer

    @staticmethod
    def backword(flow_data, layer, config):
        """
        反向传播
        :param flow_data: 流动数据
        :param layer: 层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param config:配置
        :return: 流动数据， 更新后的层
        """
        flow_data = flow_data.T
        # 变化为和前向传播时输入的尺寸一样
        flow_data = flow_data.reshape(layer['input'].shape[0], layer['input'].shape[1], layer['input'].shape[2],
                                      layer['input'].shape[3])
        return flow_data, layer

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
