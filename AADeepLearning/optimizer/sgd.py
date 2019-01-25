
class Sgd:
    """
    批量梯度下降
    """
    @staticmethod
    def update_parameters(layer, keys, learning_rate):
        """
        更新参数
        :param layer:  层，包含该层的权重、偏置项、梯度、前向输入输出缓存、实例化对象等信息
        :param keys: 需要更新的键名
        :param learning_rate: 学习率
        :return:  更新后的层
        """
        for key in keys:
            layer[key] -= learning_rate * layer["d" + key]
        return layer
