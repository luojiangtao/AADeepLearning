"""
预测下一个字母
"""
from __future__ import print_function
import numpy as np

np.random.seed(1337)
# 生产环境
# from aa_deep_learning.aadeeplearning.aadeeplearning_old import AADeepLearning
from aa_deep_learning.AADeepLearning import AADeepLearning

# print(y_test.shape)
# 网络配置文件
config = {
    # 初始学习率
    "learning_rate": 0.001,
    # 学习率衰减: 通常设置为 0.99
    "learning_rate_decay": 0.9999,
    # 优化策略: sgd/momentum/rmsprop
    "optimizer": "momentum",
    # 使用动量的梯度下降算法做优化,可以设置这一项，默认值为 0.9 ，一般不需要调整
    "momentum_coefficient": 0.95,
    # rmsprop优化器的衰减系数
    "rmsprop_decay": 0.95,
    # 正则化系数
    "reg_coefficient": 0,
    # 训练多少次
    "number_iteration": 10000,
    # 每次用多少个样本训练
    "batch_size": 2,
    # 每隔几个迭代周期评估一次准确率？
    "evaluate_interval": 10,




    # 每隔几个迭代周期保存一次快照？

    # 是否以fine_tune方式训练？ true/false

    # 预训练参数模型所在路径
    "pre_train_model": "./iter5.gordonmodel"
}

net = [
    {
        # 层名
        "name": "lstm_1",
        # 层类型
        "type": "lstm",
        # 神经元个数
        "neurons_number": 50,
        # 权重初始化方式  msra/xavier/gaussian/xavier
        "weight_init": "xavier"
    }
    ,
    {
        # 层名
        "name": "softmax",
        # 层类型
        "type": "softmax"
    }
]


def str2onehot(str):
    word2int = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10
        , 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19,
                'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}
    int2word = {v: k for k, v in word2int.items()}
    element_vector = [0] * 26
    element_vector[word2int[str]] = 1
    return element_vector


def onehot2str(element_vector):
    int2word = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i',
                9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r',
                18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
    i = np.argmax(element_vector)
    return int2word[i]


def list2vector(x, y):
    x_vector = np.zeros((len(x), len(x[0]), 26))
    y_vector = np.zeros((len(y), 26))
    for i, value in enumerate(x):
        j = 0
        for letter in value:
            x_vector[i][j] = str2onehot(letter)
            j = j + 1
        y_vector[i] = str2onehot(y[i])
    return x_vector, y_vector

# x = ['abc', 'bcd', 'cde', 'def', 'efg']
# y = ['d', 'e', 'f', 'g', 'h']

x = ['abc', 'bcd', 'cde', 'def', 'efg', 'fgh', 'ghi', 'hij', 'ijk', 'jkl', 'klm', 'lmn', 'mno', 'nop', 'opq', 'pqr',
     'qrs', 'rst', 'stu', 'tuv', 'uvw', 'vwx', 'wxy', 'xyz', 'yza', 'zab']
y = ['d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
     'a', 'b', 'c']

x_train, y_train = list2vector(x, y)

x = ['cde']
y = ['f']
x_test, y_test = list2vector(x, y)
print("x_train.shape:",x_train.shape)
print("y_train.shape:",y_train.shape)
print("x_test.shape:",x_test.shape)
print("y_test.shape:",y_test.shape)

AA = AADeepLearning(net=net, config=config)
AA.train(x_train=x_train, y_train=y_train)

result, accuracy = AA.predict(x_test=x_test, y_test=y_test)
print("accuracy:", accuracy)

print("test letter: " + x[0])
print("true letter: " + y[0])
print("predict letter: " + onehot2str(result[0]))
