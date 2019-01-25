import numpy as np

# wx = np.random.randn(5, 3, 3)
# b = np.ones((5, 1, 1))
#
# print(wx)
# print(b)
#
# print(wx+b)

a = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 10, 9]])
# a = np.array([[0, 0, 0],
#                [0, 0, 0],
#                [0, 0, 0]])
# print(v3.shape)
# print(v3[0])
# print(v3[0][0])
# print(v3[0:1, 0:2])  # 逗号前选取行（前取后不取），逗号后选取列
# print(v3[1:3, 0:2])  # 逗号前选取行（前取后不取），逗号后选取列
# b=np.array(a >= a.max())*np.ones(a.shape)
a=np.pad(a, ((2,2), (2,2)), 'constant')
a=np.pad(a, ((2,2), (2,2)), 'constant')

# print(np.unravel_index(a.argmax(), a.shape))
print(a)
