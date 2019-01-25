"""Training-related part of the Keras engine.
"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import warnings
# import copy
import time
import pickle
import numpy as np
from .layer.fully_connected import FullyConnected
from .layer.dropout import Dropout
from .layer.batch_normalization import BatchNormalization
from .layer.rnn import RNN
from .layer.lstm import LSTM
from .activation.relu import Relu
from .activation.sigmoid import Sigmoid
from .activation.tanh import Tanh
from .loss.softmax import SoftMax
from .loss.svm import SVM
from .layer.convolutional import Convolutional
from .layer.pooling import Pooling
from .layer.flatten import Flatten
from .config import get_default_config


class AADeepLearning:
    """
    入口
    """
    config = None
    # 损失值
    loss = []
    # 训练数据 shape: (60000, 28, 28, 1)  (样本数, 宽, 高, 通道数)
    train_data = []
    # 训练数据标签
    train_label = []
    # 损失值
    test_data = []
    # 损失值
    test_lable = []
    # 损失值
    input_shape = 0
    # 学习率
    learning_rate = 0
    # 神经网络层数
    layer_number = 0
    # 神经网络参数 weight和bias
    net = {}
    # 缓存loss
    loss_list = []
    # 缓存准确率
    accuracy_list = []

    def __init__(self, net={}, config={}):
        """
        初始化
        :param net: 网络结构
        :param config: 配置项
        """
        # 合并配置文件，后者覆盖前者
        self.config = {**get_default_config(), **config}
        # 网络结构和定义层一致
        self.net = net
        self.learning_rate = self.config['learning_rate']
        self.net = self.init_net(net)
        self.is_load_model = False
        if self.config["load_model"] != "":
            # 加载模型，进行预测或者继续训练
            self.reload(self.config["load_model"])
            self.is_load_model = True

    def init_net(self, net):
        """
        初始化网络所需的对象，方便后期调用，不用每次都重复判断
        :param net: 网络结构
        :return: 网络结构
        """
        for i, layer in enumerate(net):
            if layer['type'] == 'convolutional':
                net[i]['object'] = Convolutional()
            elif layer['type'] == 'pooling':
                net[i]['object'] = Pooling()
            elif layer['type'] == 'flatten':
                net[i]['object'] = Flatten()
            elif layer['type'] == 'fully_connected':
                net[i]['object'] = FullyConnected()
            elif layer['type'] == 'dropout':
                net[i]['object'] = Dropout()
            elif layer['type'] == 'batch_normalization':
                net[i]['object'] = BatchNormalization()
            elif layer['type'] == 'relu':
                net[i]['object'] = Relu()
            elif layer['type'] == 'sigmoid':
                net[i]['object'] = Sigmoid()
            elif layer['type'] == 'tanh':
                net[i]['object'] = Tanh()
            elif layer['type'] == 'rnn':
                net[i]['object'] = RNN()
            elif layer['type'] == 'lstm':
                net[i]['object'] = LSTM()
            elif layer['type'] == 'softmax':
                net[i]['object'] = SoftMax()
            elif layer['type'] == 'svm':
                net[i]['object'] = SVM()
        return net

    def train(self, x_train=None, y_train=None, is_train=True):
        """
        训练
        :param x_train: 数据
        :param y_train: 标签
        :param is_train: 是否是训练模式
        """
        if len(x_train.shape) == 4:
            # 训练立方体数据 例如图片数据  宽*高*通道数
            flow_data_shape = {
                "batch_size": self.config['batch_size'],
                "channel": x_train.shape[1],
                "height": x_train.shape[2],
                "width": x_train.shape[3]
            }
        else:
            # 训练序列数据 样本 * 序列个数 * 序列长度
            flow_data_shape = {
                "batch_size": self.config['batch_size'],
                "sequence_number": x_train.shape[1],
                "sequence_length": x_train.shape[2]
            }
        # 1，初始化网络参数
        if self.is_load_model == False:
            # 没有载入已训练好的模型，则初始化
            self.net = self.init_parameters(flow_data_shape)
        for iteration in range(1, self.config['number_iteration'] + 1):
            x_train_batch, y_train_batch = self.next_batch(x_train, y_train, self.config['batch_size'])
            # 2，前向传播
            flow_data = self.forward_pass(self.net, x_train_batch, is_train=is_train)
            # loss = self.compute_cost(flow_data, y_train_batch)
            # 3，调用最后一层的计算损失函数，计算损失
            loss = self.net[len(self.net)-1]['object'].compute_cost(flow_data, self.net[len(self.net)-1], y_train_batch)
            self.loss_list.append(loss)
            # 4，反向传播，求梯度
            self.net = self.backward_pass(self.net, flow_data, y_train_batch)
            # 梯度检验
            # self.gradient_check(x=x_train_batch, y=y_train_batch, net=self.net, layer_name='convolutional_1', weight_key='W', gradient_key='dW')
            # exit()
            # 5，根据梯度更新一次参数
            self.net = self.update_parameters(self.net, iteration)
            if iteration % self.config["display"] == 0:
                # self.check_weight(self.net)
                _, accuracy = self.predict(x_train_batch, y_train_batch, is_train=is_train)
                self.accuracy_list.append(accuracy)
                now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print(now_time, '   iteration:', iteration, '   loss:', loss, ' accuracy:', accuracy)
            if self.config["save_model"] != "" and iteration % self.config["save_iteration"] == 0:
                print('saving model...')
                self.save(self.config["save_model"] + "-" + str(iteration) + '.model')


    def init_parameters(self, flow_data_shape):
        """
        初始化权重和偏置项
        :param flow_data_shape: 流动数据形状
        :return: 网络结构
        """
        net = self.net
        for i, layer in enumerate(net):
            net[i], flow_data_shape = layer['object'].init(layer=layer, flow_data_shape=flow_data_shape,
                                                           config=self.config)
        return net

    def forward_pass(self, net, x, is_train=False):
        """
        前向传播
        :param net: 网络结构
        :param x: 数据
        :param is_train: 是否是训练模式
        :return: 流动数据
        """
        # 流动数据，一层一层的计算，并向后流动
        flow_data = x
        for i, layer in enumerate(net):
            # 缓存当前层的输入
            net[i]['input'] = flow_data
            flow_data, net[i] = layer["object"].forword(flow_data=flow_data, layer=layer, is_train=is_train)
            # 缓存当前层的输出
            net[i]['output'] = flow_data
        return flow_data

    def backward_pass(self, net, flow_data, train_label):
        """
        反向传播
        :param net: 网络结构
        :param flow_data:  前向传播最后一层输出
        :param train_label:  标签
        :return: 包含梯度的网络结构
        """
        layer_number = len(net)
        for i in reversed(range(0, layer_number)):
            layer = net[i]
            if i == len(net)-1:
                # 最后一层
                flow_data = layer["object"].backword(flow_data=flow_data, layer=layer, label=train_label)
            else:
                flow_data, net[i] = layer["object"].backword(flow_data=flow_data, layer=layer, config=self.config)
        return net

    def update_parameters(self, net, iteration):
        """
        更新权重，偏置项
        :param net: 网络结构
        :param iteration: 迭代次数
        :return: 更新权重，偏置项后的网络结构
        """
        for i, layer in enumerate(net):
            net[i] = layer['object'].update_parameters(layer=layer, config=self.config, iteration=iteration)
        return net

    def save(self, path="AA.model"):
        """
        保存模型
        :param path: 路径
        """
        with open(path, "wb") as f:
            pickle.dump(self.net, f)

    def reload(self, path="AA.model"):
        """
        载入模型
        :param path: 路径
        """
        with open(path, "rb") as f:
            self.net = pickle.load(f)

    def predict(self, x_test=None, y_test=None, is_train=False):
        """
        预测
        :param x_test: 预测数据
        :param y_test: 预测标签
        :param is_train: 是否是训练模式
        :return: 概率分布矩阵，准确率
        """
        # if x_test.shape[0] > 500:
        #     print("Verify the accuracy on " + str(x_test.shape[0]) + " test set, please wait a moment.")
        flow_data = self.forward_pass(self.net, x_test, is_train)
        flow_data = np.array(flow_data).T
        batch_size = y_test.shape[0]
        right = 0
        for i in range(0, batch_size):
            index = np.argmax(flow_data[i])
            if y_test[i][index] == 1:
                right += 1
        accuracy = right / batch_size
        return flow_data, accuracy

    def next_batch(self, train_data, train_label, batch_size):
        """
        随机获取下一批数据
        :param train_data:
        :param train_label:
        :param batch_size:
        :return:
        """
        index = [i for i in range(0, len(train_label))]
        # 洗牌后卷积核个数居然会改变固定位置的图片？
        np.random.shuffle(index)
        batch_data = []
        batch_label = []
        for i in range(0, batch_size):
            batch_data.append(train_data[index[i]])
            batch_label.append(train_label[index[i]])
        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        return batch_data, batch_label

    def visualization_loss(self):
        """
        画出损失曲线
        :return:
        """
        import matplotlib.pyplot as plt
        plt.plot(self.loss_list, 'r')
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.show()

    def visualization_accuracy(self):
        """
        画出正确率曲线
        :return:
        """
        import matplotlib.pyplot as plt
        plt.plot(self.accuracy_list, 'g')
        plt.xlabel("display")
        plt.ylabel("accuracy")
        plt.show()

    def check_weight(self, net):
        """
        检查权重，查看小于1e-8的比例
        :param net:
        :return:
        """
        for i, layer in enumerate(net):
            if layer['type'] == 'fully_connected':
                print(layer["name"], ":dW|<1e-8 :", np.sum(abs(layer['dW']) < 1e-8), "/",
                      layer['dW'].shape[0] * layer['dW'].shape[1])
                print(layer['name'] + ":db|<1e-8 :", np.sum(abs(layer['db']) < 1e-8), "/",
                      layer['db'].shape[0] * layer['db'].shape[1])
            elif layer['type'] == 'convolutional':
                print(layer["name"], ":dW|<1e-8 :", np.sum(abs(layer['dW']) < 1e-8), "/",
                      layer['dW'].shape[0] * layer['dW'].shape[1] * layer['dW'].shape[2] * layer['dW'].shape[3])
                print(layer['name'] + ":db|<1e-8 :", np.sum(abs(layer['db']) < 1e-8), "/",
                      layer['db'].shape[0] * layer['db'].shape[1] * layer['db'].shape[2])
            elif layer['type'] == 'rnn':
                print(layer['name'] + ":weight_U_gradient" + str(i) + "|<1e-8 :",
                      np.sum(abs(layer['weight_U_gradient']) < 1e-8), "/",
                      layer['weight_U_gradient'].shape[0] * layer['weight_U_gradient'].shape[1])
                print(layer['name'] + ":weight_W_gradient" + str(i) + "|<1e-8 :",
                      np.sum(abs(layer['weight_W_gradient']) < 1e-8), "/",
                      layer['weight_W_gradient'].shape[0] * layer['weight_W_gradient'].shape[1])
                print(layer['name'] + ":weight_V_gradient" + str(i) + "|<1e-8 :",
                      np.sum(abs(layer['weight_V_gradient']) < 1e-8), "/",
                      layer['weight_V_gradient'].shape[0] * layer['weight_V_gradient'].shape[1])
            elif layer['type'] == 'lstm':
                print(layer['name'] + ":dWf" + str(i) + "|<1e-8 :", np.sum(abs(layer['dWf']) < 1e-8), "/",
                      layer['dWf'].shape[0] * layer['dWf'].shape[1])
                print(layer['name'] + ":dUf" + str(i) + "|<1e-8 :", np.sum(abs(layer['dUf']) < 1e-8), "/",
                      layer['dUf'].shape[0] * layer['dUf'].shape[1])
                print(layer['name'] + ":dbf" + str(i) + "|<1e-8 :", np.sum(abs(layer['dbf']) < 1e-8), "/",
                      layer['dbf'].shape[0] * layer['dbf'].shape[1])

                print(layer['name'] + ":dWi" + str(i) + "|<1e-8 :", np.sum(abs(layer['dWi']) < 1e-8), "/",
                      layer['dWi'].shape[0] * layer['dWi'].shape[1])
                print(layer['name'] + ":dUf" + str(i) + "|<1e-8 :", np.sum(abs(layer['dUf']) < 1e-8), "/",
                      layer['dUi'].shape[0] * layer['dUi'].shape[1])
                print(layer['name'] + ":dbf" + str(i) + "|<1e-8 :", np.sum(abs(layer['dbf']) < 1e-8), "/",
                      layer['dbi'].shape[0] * layer['dbi'].shape[1])

    def gradient_check(self, x, y, net, layer_name, weight_key, gradient_key, epsilon=1e-4):
        """
        梯度检验
        :param x: 数据
        :param y: 标签
        :param net: 网络结构
        :param layer_name: 需要检验的层名称
        :param weight_key: 需要检验的权重键名
        :param gradient_key: 需要检验的梯度键名
        :param epsilon: 数值逼近的x长度
        """
        # 1,要检验的梯度展成一列
        layer_number = -1  # 第几层
        for j, layer in enumerate(net):
            if layer['name'] == layer_name:
                layer_number = j
                break
        assert layer_number != -1
        # 梯度字典转列向量(n,1)
        gradient_vector = np.reshape(net[layer_number][gradient_key], (-1, 1))
        # 参数字典转列向量(n,1)
        weight_vector = np.reshape(net[layer_number][weight_key], (-1, 1))

        # 数值逼近求得的梯度
        gradient_vector_approach = np.zeros(gradient_vector.shape)
        lenght = weight_vector.shape[0]
        # 遍历，每次求权重一个数据点的梯度，然后串联起来
        for i in range(lenght):
            if i % 10 == 0:
                print("gradient checking i/len=", i, "/", lenght)
            weight_vector_plus = np.copy(weight_vector)
            weight_vector_plus[i][0] = weight_vector_plus[i][0] + epsilon
            net[layer_number][weight_key] = np.reshape(weight_vector_plus, net[layer_number][weight_key].shape)
            # 2，前向传播
            flow_data = self.forward_pass(net=net, x=x)
            # 3，计算损失
            # J_plus_epsilon = self.compute_cost(flow_data, y)
            J_plus_epsilon = net[len(net) - 1]['object'].compute_cost(flow_data, net[len(net) - 1], y)

            weight_vector_minus = np.copy(weight_vector)
            weight_vector_minus[i][0] = weight_vector_minus[i][0] - epsilon
            net[layer_number][weight_key] = np.reshape(weight_vector_minus, net[layer_number][weight_key].shape)
            # 2，前向传播
            flow_data = self.forward_pass(net=net, x=x)
            # 3，计算损失
            # J_minus_epsilon = self.compute_cost(flow_data, y)
            J_minus_epsilon = net[len(net) - 1]['object'].compute_cost(flow_data, net[len(net) - 1], y)

            # 数值逼近求得梯度
            gradient_vector_approach[i][0] = (J_plus_epsilon - J_minus_epsilon) / (epsilon * 2)

        # 和解析解求得的梯度做欧式距离
        diff = np.sqrt(np.sum((gradient_vector - gradient_vector_approach) ** 2)) / (
                    np.sqrt(np.sum((gradient_vector) ** 2)) + np.sqrt(np.sum((gradient_vector_approach) ** 2)))
        # 错误阈值
        if diff > 1e-4:
            print("Maybe a mistake in your bakeward pass!!!  diff=", diff)
        else:
            print("No problem in your bakeward pass!!!  diff=", diff)
