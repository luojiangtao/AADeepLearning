"""
预测下一个字母
"""
import numpy as np
np.random.seed(1337)

from aa_deep_learning.AADeepLearning import AADeepLearning

# 网络配置文件
config = {
    # 初始学习率
    "learning_rate": 0.001,
    # 优化策略: sgd/momentum/rmsprop/adam
    "optimizer": "sgd",
    # 训练多少次
    "number_iteration": 5000,
    # 每次用多少个样本训练
    "batch_size": 2,
    # 迭代多少次打印一次信息
    "display": 500,
}

net = [
    {
        # 层名
        "name": "rnn_1",
        # 层类型，循环神经网络层 目前只支持 n->1 输出
        "type": "rnn",
        # 神经元个数
        "neurons_number": 128,
        # 权重初始化方式  msra/xavier/gaussian
        "weight_init": "msra"
    },
    {
        # 层名
        "name": "softmax",
        # 层类型
        "type": "softmax"
    }
]


def str2onehot(str):
    """
    字符串转 one hot 向量
    :param str: 字符串
    :return: one hot 向量
    """
    word2int = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10
        , 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19,
                'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}
    element_vector = [0] * 26
    element_vector[word2int[str]] = 1
    return element_vector


def onehot2str(element_vector):
    """
     one hot 向量转字符串
    :param element_vector:one hot 向量
    :return:  字符串
    """
    int2word = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i',
                9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r',
                18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
    i = np.argmax(element_vector)
    return int2word[i]


def list2vector(x, y):
    """
    列表转one hot 向量
    :param x: 数据
    :param y: 标签
    :return: one hot 向量
    """
    x_vector = np.zeros((len(x), len(x[0]), 26))
    y_vector = np.zeros((len(y), 26))
    for i, value in enumerate(x):
        j = 0
        for letter in value:
            x_vector[i][j] = str2onehot(letter)
            j = j + 1
        y_vector[i] = str2onehot(y[i])
    return x_vector, y_vector

x = ['abc', 'bcd', 'cde', 'def', 'efg', 'fgh', 'ghi', 'hij', 'ijk', 'jkl', 'klm', 'lmn', 'mno', 'nop', 'opq', 'pqr',
     'qrs', 'rst', 'stu', 'tuv', 'uvw', 'vwx', 'wxy', 'xyz', 'yza', 'zab']
y = ['d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
     'a', 'b', 'c']
# 训练数据
x_train, y_train = list2vector(x, y)

# 预测数据
x = ['bcd']
y = ['e']
x_test, y_test = list2vector(x, y)

print("x_train.shape:",x_train.shape)
print("y_train.shape:",y_train.shape)
print("x_test.shape:",x_test.shape)
print("y_test.shape:",y_test.shape)

# 定义模型，传入网络结构和配置项
AA = AADeepLearning(net=net, config=config)
# 训练模型
AA.train(x_train=x_train, y_train=y_train)

# 使用测试集预测，返回概率分布和准确率， score:样本在各个分类上的概率， accuracy:准确率
result, accuracy = AA.predict(x_test=x_test, y_test=y_test)
print("accuracy:", accuracy)

print("test letter: " + x[0])
print("true letter: " + y[0])
print("predict letter: " + onehot2str(result[0]))
