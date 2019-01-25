# lstm 识别 mnist 数据集
# pip install AADeepLearning
from aa_deep_learning.AADeepLearning import AADeepLearning
from aa_deep_learning.AADeepLearning.datasets import mnist
from aa_deep_learning.AADeepLearning.datasets import np_utils

# mnist数据集已经被划分成了60,000个训练集，10,000个测试集的形式，如果数据不存在则自动下载
# x_train，x_test 第一个维度是样本数目，第二维度是高，第三个维度是宽
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 将x_train, x_test的数据格式转为float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 归一化，将值映射到 0到1区间
x_train /= 255
x_test /= 255

# 因为是10分类，所以将类别向量(从0到10的整数向量)映射为二值类别矩阵，相当于将向量用one-hot重新编码
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# 网络配置文件
config = {
    # 初始学习率
    "learning_rate": 0.001,
    # 优化策略: sgd/momentum/rmsprop/adam
    "optimizer": "adam",
    # 训练多少次
    "number_iteration": 2000,
    # 每次用多少个样本训练
    "batch_size": 64,
    # 迭代多少次打印一次信息
    "display": 100,
}

# 网络结构，数据将从上往下传播
net = [
    {
        # 层名
        "name": "lstm_1",
        # 层类型，lstm网络中，图片高作为序列个数，图片宽作为序列长度， 暂时只支持 N->1 模式
        "type": "lstm",
        # 神经元个数
        "neurons_number": 60,
        # 权重初始化方式  msra/gaussian/xavier
        "weight_init": "xavier"
    },
    {
        # 层名
        "name": "relu_1",
        # 层类型
        "type": "relu"
    },
    {
        # 层名
        "name": "fully_connected_2",
        # 层类型，全连接层，
        "type": "fully_connected",
        # 神经元个数， 因为是10分类，所以神经元个数为10
        "neurons_number": 10,
        # 权重初始化方式  msra/xavier/gaussian
        "weight_init": "msra"
    },
    {
        # 层名
        "name": "softmax_1",
        # 层类型，分类层，最终输出十分类的概率分布
        "type": "softmax"
    }
]

# 定义模型，传入网络结构和配置项
AA = AADeepLearning(net=net, config=config)
# 训练模型
AA.train(x_train=x_train, y_train=y_train)

# 使用测试集预测，返回概率分布和准确率， score:样本在各个分类上的概率， accuracy:准确率
score, accuracy = AA.predict(x_test=x_test, y_test=y_test)
print("test set accuracy:", accuracy)
