# 加载模型，直接预测
# pip install AADeepLearning
from aa_deep_learning.AADeepLearning import AADeepLearning
from aa_deep_learning.AADeepLearning.datasets import mnist
from aa_deep_learning.AADeepLearning.datasets import np_utils

# mnist数据集已经被划分成了60,000个训练集，10,000个测试集的形式，如果数据不存在则自动下载
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 第一个维度是样本数目，第二维度是通道数表示颜色通道数，第三维度是高，第四个维度是宽
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

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
    # 参数模型所在路径，不为空框架会加载模型，用于预测或继续训练
    "load_model": "AA-1000.model"
}

# 定义模型，传入配置项
AA = AADeepLearning(config=config)

# 使用测试集预测，返回概率分布和准确率， score:样本在各个分类上的概率， accuracy:准确率
score, accuracy = AA.predict(x_test=x_test, y_test=y_test)
print("test set accuracy:", accuracy)
