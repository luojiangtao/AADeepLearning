from __future__ import print_function
import numpy as np

np.random.seed(1337)
# 生产环境
from aa_deep_learning.AADeepLearning import AADeepLearning
from aa_deep_learning.AADeepLearning.datasets import cifar10
from aa_deep_learning.AADeepLearning.datasets import cifar100
from aa_deep_learning.AADeepLearning.datasets import imdb
from aa_deep_learning.AADeepLearning.datasets import reuters
from aa_deep_learning.AADeepLearning.datasets import fashion_mnist
from aa_deep_learning.AADeepLearning.datasets import np_utils

from aa_deep_learning.AADeepLearning.datasets import mnist
# mnist数据集已经被划分成了60,000个训练集，10,000个测试集的形式，如果数据不存在则自动下载
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 画出minist 数字
fig = plt.figure()
plt.imshow(x_test[0],cmap = 'binary')#黑白显示
plt.show()

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = np.transpose(x_train, (0,2,3,1))
# plt.figure(figsize=(1,1))
# plt.imshow(x_train[0])
# plt.show()

# (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
# x_train = np.transpose(x_train, (0,2,3,1))
# plt.figure(figsize=(1,1))
# plt.imshow(x_train[0])
# plt.show()

# (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
#                                                       num_words=None,
#                                                       skip_top=0,
#                                                       maxlen=None,
#                                                       seed=113,
#                                                       start_char=1,
#                                                       oov_char=2,
#                                                       index_from=3)
# print(x_train[8])

# (x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
#                                                          num_words=None,
#                                                          skip_top=0,
#                                                          maxlen=None,
#                                                          test_split=0.2,
#                                                          seed=113,
#                                                          start_char=1,
#                                                          oov_char=2,
#                                                          index_from=3)
# print(x_train[8])

# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# fig = plt.figure()
# plt.imshow(x_train[10],cmap = 'binary')#黑白显示
# plt.show()

# (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print(x_train[8])

# print('x_train shape:', x_train.shape)
# print('y_train shape:', y_train.shape)
# print('x_test shape:', x_test.shape)
# print('y_test shape:', y_test.shape)
